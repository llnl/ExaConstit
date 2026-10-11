#include "utilities/unified_logger.hpp"

#include "postprocessing/postprocessing_file_manager.hpp"

#include <iomanip> // For std::put_time

#include <sys/select.h> // For select() - monitoring file descriptors

namespace exaconstit {

// ============================================================================
// STATIC MEMBER INITIALIZATION
// ============================================================================
// These must be defined in exactly one translation unit
std::unique_ptr<UnifiedLogger> UnifiedLogger::instance = nullptr;
std::mutex UnifiedLogger::instance_mutex;

// ============================================================================
// SINGLETON ACCESS
// ============================================================================
UnifiedLogger& UnifiedLogger::get_instance() {
    // Double-checked locking pattern for thread-safe lazy initialization
    // First check without lock (fast path)
    if (!instance) {
        // Acquire lock and check again (slow path)
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (!instance) {
            // Create instance - using private constructor via friendship
            instance.reset(new UnifiedLogger());
        }
    }
    return *instance;
}

// ============================================================================
// INITIALIZATION
// ============================================================================
void UnifiedLogger::initialize(const ExaOptions& options) {
    PostProcessingFileManager file_manager(options);
    // Step 1: Set up log directory path
    // Use file manager's structure if available for consistency
    // The output directory was already resolved once on rank 0 and broadcast to all
    // ranks in ExaOptions::parse_options, so it is used as is rather than resolving any
    // symlinks in it again here.
    fs::path base_dir = file_manager.GetOutputDirectory();
    log_directory = base_dir / "logs";
    // Step 2: Create directory structure
    if (mpi_rank == 0) {
        // No exists() check first: create_directories() is a no-op if the directory
        // already exists, and checking first only opens a window for the path to change
        // between the check and the creation.
        std::error_code ec;
        fs::create_directories(log_directory, ec);
        if (ec) {
            std::cerr << "Warning: Failed to create log directory: " << ec.message() << std::endl;
        } else if (!fs::is_directory(log_directory, ec)) {
            std::cerr << "Error: Log path exists but is not a directory: " << log_directory
                      << std::endl;
        }
    }

    // Synchronize all ranks before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Set up main log file path
    main_log_filename = log_directory / (options.basename + "_simulation.log");

    // Step 4: Open main log file
    // All ranks append to same file (OS handles concurrent writes)
    main_log_file = std::make_unique<std::ofstream>(main_log_filename, std::ios::app);

    if (!main_log_file->is_open()) {
        std::cerr << "Warning: Failed to open main log file: " << main_log_filename << std::endl;
        return;
    }

    // Step 5: Enable tee functionality
    // From this point, all cout/cerr goes to both terminal and file
    enable_main_logging();

    // Step 6: Write header (rank 0 only to avoid duplication)
    if (mpi_rank == 0) {
        std::cout << "\n=== ExaConstit Simulation: " << options.basename << " ===" << std::endl;
        std::cout << "MPI Ranks: " << mpi_size << std::endl;
        std::cout << "Log Directory: " << log_directory << std::endl;
        std::cout << "Main Log: " << main_log_filename.filename() << std::endl;

        // Add timestamp using C++17 time formatting
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "Start Time: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                  << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
}

// ============================================================================
// TEE MODE - MAIN LOGGING
// ============================================================================
void UnifiedLogger::enable_main_logging() {
    if (!main_log_file || !main_log_file->is_open()) {
        return;
    }

    // Set up cout tee
    if (!cout_guard) {
        // Get cout's current streambuf (this is what writes to terminal)
        std::streambuf* original_buf = std::cout.rdbuf();
        // Create guard to restore it later
        cout_guard.emplace(std::cout);
        // Create tee that writes to BOTH original buffer AND file
        cout_tee = std::make_unique<TeeStreambuf>(original_buf, main_log_file.get());
        // Now replace cout's buffer with our tee
        std::cout.rdbuf(cout_tee.get());
    }

    // Set up cerr tee
    if (!cerr_guard) {
        std::streambuf* original_buf = std::cerr.rdbuf();
        cerr_guard.emplace(std::cerr);
        cerr_tee = std::make_unique<TeeStreambuf>(original_buf, main_log_file.get());
        std::cerr.rdbuf(cerr_tee.get());
    }

    // Set up mfem::out tee
    if (!mfem_out_guard) {
        // MFEM's out stream might be using cout's buffer or its own
        std::streambuf* original_buf = mfem::out.rdbuf();
        if (original_buf) { // mfem::out might be disabled
            mfem_out_guard.emplace(mfem::out);
            mfem_out_tee = std::make_unique<TeeStreambuf>(original_buf, main_log_file.get());
            mfem::out.rdbuf(mfem_out_tee.get());
        }
    }

    // Set up mfem::err tee
    if (!mfem_err_guard) {
        std::streambuf* original_buf = mfem::err.rdbuf();
        if (original_buf) { // mfem::err might be disabled
            mfem_err_guard.emplace(mfem::err);
            mfem_err_tee = std::make_unique<TeeStreambuf>(original_buf, main_log_file.get());
            mfem::err.rdbuf(mfem_err_tee.get());
        }
    }
}

void UnifiedLogger::disable_main_logging() {
    // RAII guards automatically restore original buffers when reset
    cout_guard.reset(); // Restores original cout buffer
    cerr_guard.reset(); // Restores original cerr buffer
    mfem_out_guard.reset();
    mfem_err_guard.reset();

    // Clean up tee buffers
    cout_tee.reset();
    cerr_tee.reset();
    mfem_out_tee.reset();
    mfem_err_tee.reset();
}

// ============================================================================
// CAPTURE MODE - READER THREAD
// ============================================================================
void UnifiedLogger::reader_thread_func(CaptureContext* ctx) {
    // This function runs in a separate thread during capture
    // Its job: read from pipe and accumulate in stringstream

    constexpr size_t BUFFER_SIZE = 65536; // 64KB buffer
    std::vector<char> buffer(BUFFER_SIZE);

    // Main reading loop
    while (!ctx->stop_reading.load()) {
        // Use select() for efficient I/O monitoring
        // select() tells us when data is available without blocking
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(ctx->capture_pipe.read_end().get(), &read_fds);

        // Set timeout so we periodically check stop_reading
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 10000; // 10ms - more responsive than 50ms

        // Wait for data or timeout
        int ret = select(
            ctx->capture_pipe.read_end().get() + 1, &read_fds, nullptr, nullptr, &timeout);

        if (ret > 0 && FD_ISSET(ctx->capture_pipe.read_end().get(), &read_fds)) {
            // Data is available - read as much as possible
            while (true) {
                ssize_t bytes_read = ::read(
                    ctx->capture_pipe.read_end().get(), buffer.data(), buffer.size() - 1);

                if (bytes_read > 0) {
                    // Successfully read data
                    buffer[static_cast<size_t>(bytes_read)] = '\0'; // Null terminate
                    ctx->captured_output << buffer.data();
                    ctx->has_content = true;
                    ctx->bytes_captured += static_cast<size_t>(bytes_read);
                } else if (bytes_read == 0) {
                    // EOF - pipe closed
                    break;
                } else {
                    // Error or would block
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // No more data available right now
                        break;
                    }
                    // Real error - stop reading
                    break;
                }
            }
        }
        // If select times out, loop continues to check stop_reading
    }

    // Final flush - read any remaining data
    while (true) {
        ssize_t bytes_read = ::read(
            ctx->capture_pipe.read_end().get(), buffer.data(), buffer.size() - 1);
        if (bytes_read <= 0)
            break;

        buffer[static_cast<size_t>(bytes_read)] = '\0';
        ctx->captured_output << buffer.data();
        ctx->has_content = true;
        ctx->bytes_captured += static_cast<size_t>(bytes_read);
    }
}

// ============================================================================
// GPU OUTPUT SYNCHRONIZATION
// ============================================================================
void UnifiedLogger::flush_gpu_output() {
    // GPU printf uses device-side buffers that must be explicitly flushed

#ifdef RAJA_ENABLE_CUDA
    // CUDA: Force device synchronization
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Don't print error - we might be capturing output
        // Store error info in capture context if needed
    }
#elif defined(RAJA_ENABLE_HIP)
    // HIP: Force device synchronization
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        // Don't print error - we might be capturing output
    }
#endif

    // Additional delay to ensure driver flushes printf buffers
    // This is unfortunately necessary for reliable GPU printf capture
    usleep(5000); // 5ms
}

// ============================================================================
// BEGIN CAPTURE
// ============================================================================
void UnifiedLogger::begin_capture(const std::string& filename, bool suppress_non_zero_ranks) {
    std::lock_guard<std::mutex> lock(capture_mutex);

    // CRITICAL: Flush all C++ streams before redirecting
    std::cout.flush();
    std::cerr.flush();
    mfem::out.flush();
    mfem::err.flush();

    // Also flush C stdio buffers
    fflush(stdout);
    fflush(stderr);

    // Temporarily disable tee functionality during capture
    if (!in_capture_mode && (cout_tee || cerr_tee || mfem_out_tee || mfem_err_tee)) {
        in_capture_mode = true;

        // Use RAII guards to temporarily restore original buffers
        if (cout_guard && cout_tee) {
            temp_cout_guard.emplace(std::cout);
            temp_cout_guard->set_buffer(cout_guard->get_original());
        }
        if (cerr_guard && cerr_tee) {
            temp_cerr_guard.emplace(std::cerr);
            temp_cerr_guard->set_buffer(cerr_guard->get_original());
        }
        if (mfem_out_guard && mfem_out_tee) {
            temp_mfem_out_guard.emplace(mfem::out);
            temp_mfem_out_guard->set_buffer(mfem_out_guard->get_original());
        }
        if (mfem_err_guard && mfem_err_tee) {
            temp_mfem_err_guard.emplace(mfem::err);
            temp_mfem_err_guard->set_buffer(mfem_err_guard->get_original());
        }
    }

    // Create new capture context
    auto ctx = std::make_unique<CaptureContext>();
    ctx->output_filename = log_directory / filename;
    ctx->suppress_non_zero_ranks = suppress_non_zero_ranks;
    ctx->start_time = std::chrono::steady_clock::now();

    // Special handling for rank suppression
    if (suppress_non_zero_ranks && mpi_rank != 0) {
        // For non-zero ranks, redirect both file descriptors AND C++ streams

        // Open /dev/null for file descriptors
        FileDescriptor devnull(::open("/dev/null", O_WRONLY));
        if (!devnull) {
            throw std::system_error(errno, std::system_category(), "Failed to open /dev/null");
        }

        // Redirect file descriptors
        ctx->stdout_dup.emplace(STDOUT_FILENO);
        ctx->stderr_dup.emplace(STDERR_FILENO);
        ctx->stdout_dup->redirect_to(devnull.get());
        ctx->stderr_dup->redirect_to(devnull.get());

        // Also redirect C++ streams to null stream using RAII
        ctx->null_stream = std::make_unique<std::ofstream>("/dev/null");
        if (ctx->null_stream->is_open()) {
            ctx->cout_null_guard.emplace(std::cout);
            ctx->cout_null_guard->set_buffer(ctx->null_stream->rdbuf());

            ctx->cerr_null_guard.emplace(std::cerr);
            ctx->cerr_null_guard->set_buffer(ctx->null_stream->rdbuf());

            ctx->mfem_out_null_guard.emplace(mfem::out);
            ctx->mfem_out_null_guard->set_buffer(ctx->null_stream->rdbuf());

            ctx->mfem_err_null_guard.emplace(mfem::err);
            ctx->mfem_err_null_guard->set_buffer(ctx->null_stream->rdbuf());
        }

        capture_stack.push(std::move(ctx));
        return;
    }

    // Normal capture setup (rest of the existing code)
    try {
        // Create pipe with non-blocking read
        ctx->capture_pipe.set_non_blocking(true, false);

        // Save and redirect file descriptors
        ctx->stdout_dup.emplace(STDOUT_FILENO);
        ctx->stderr_dup.emplace(STDERR_FILENO);
        ctx->stdout_dup->redirect_to(ctx->capture_pipe.write_end().get());
        ctx->stderr_dup->redirect_to(ctx->capture_pipe.write_end().get());

        // Start reader thread
        ctx->reader_thread = std::thread(&UnifiedLogger::reader_thread_func, this, ctx.get());

        // Push onto stack
        capture_stack.push(std::move(ctx));

    } catch (const std::exception& e) {
        // RAII will automatically restore everything
        throw std::runtime_error(std::string("Failed to begin capture: ") + e.what());
    }
}

// ============================================================================
// END CAPTURE
// ============================================================================
std::string UnifiedLogger::end_capture() {
    std::lock_guard<std::mutex> lock(capture_mutex);

    if (capture_stack.empty()) {
        return "";
    }

    // Get current capture context
    auto ctx = std::move(capture_stack.top());
    capture_stack.pop();

    // Handle suppressed ranks
    if (ctx->suppress_non_zero_ranks && mpi_rank != 0) {
        // RAII automatically restores everything when ctx goes out of scope

        // Re-enable tee if this was the last capture
        if (capture_stack.empty() && in_capture_mode) {
            restore_tee_after_capture();
        }

        return "";
    }

    // Normal capture end (existing flush code)
    flush_gpu_output();
    std::cout.flush();
    std::cerr.flush();
    mfem::out.flush();
    mfem::err.flush();
    fflush(stdout);
    fflush(stderr);

    // RAII automatically restores stdout/stderr
    ctx->stdout_dup.reset();
    ctx->stderr_dup.reset();

    // Close write end and stop reader
    ctx->capture_pipe.write_end().close();
    ctx->stop_reading.store(true);
    if (ctx->reader_thread.joinable()) {
        ctx->reader_thread.join();
    }

    // Re-enable tee if this was the last capture
    if (capture_stack.empty() && in_capture_mode) {
        restore_tee_after_capture();
    }

    // Step 5: Check capture duration for warnings
    auto duration = std::chrono::steady_clock::now() - ctx->start_time;
    if (duration > std::chrono::seconds(5)) {
        std::cerr << "[Logger] Long capture duration: "
                  << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
                  << " seconds for " << ctx->output_filename.filename() << std::endl;
    }

    // Step 6: Process captured content
    std::string content = ctx->captured_output.str();

    // Use string_view for efficient trimming (no copies)
    std::string_view sv(content);

    // Remove leading whitespace
    size_t start = sv.find_first_not_of(" \t\n\r");
    if (start == std::string_view::npos) {
        // All whitespace - no content
        stats_.total_captures++;
        return "";
    }
    sv.remove_prefix(start);

    // Remove trailing whitespace
    size_t end = sv.find_last_not_of(" \t\n\r");
    if (end != std::string_view::npos) {
        sv = sv.substr(0, end + 1);
    }

    // Update statistics
    stats_.total_captures++;

    // Step 7: Write file ONLY if content exists
    if (!sv.empty() && ctx->has_content) {
        // The log directory is built from the output directory that was resolved once in
        // ExaOptions::parse_options, so the path is used as is. Several ranks can reach
        // this at the same time, so the directory is created without a separate exists()
        // check beforehand (create_directories() is a no-op if it already exists).
        const fs::path& output_path = ctx->output_filename;
        fs::path output_dir = output_path.parent_path();

        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Warning: Failed to create output directory: " << output_dir << ": "
                      << ec.message() << std::endl;
        }

        // Open output file
        std::ofstream out_file(output_path, std::ios::app);
        if (out_file.is_open()) {
            // Write header with metadata
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);

            out_file << "\n=== Capture Session ===" << std::endl;
            out_file << "Timestamp: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                     << std::endl;
            out_file << "MPI Rank: " << mpi_rank << std::endl;
            out_file << "Duration: "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
                     << " ms" << std::endl;
            out_file << "Bytes Captured: " << ctx->bytes_captured << std::endl;
            out_file << "------------------------" << std::endl;
            out_file << sv; // Write actual content
            out_file << "\n========================\n" << std::endl;

            // Update statistics
            stats_.files_created++;
            stats_.created_files.push_back(ctx->output_filename.string());

            // Log to main that we created a file
            if (main_log_file && main_log_file->is_open()) {
                if (debugging_logging) {
                    *main_log_file
                        << "[Logger] Created output file: " << ctx->output_filename.filename()
                        << " (" << ctx->bytes_captured << " bytes)" << std::endl;
                }
            }

            return ctx->output_filename.string();
        }
    }

    // No file created
    return "";
}

void UnifiedLogger::restore_tee_after_capture() {
    if (!in_capture_mode)
        return;

    // Simply reset the temporary guards - RAII handles restoration
    temp_cout_guard.reset();
    temp_cerr_guard.reset();
    temp_mfem_out_guard.reset();
    temp_mfem_err_guard.reset();

    in_capture_mode = false;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================
std::string UnifiedLogger::get_material_log_filename(const std::string& model_type,
                                                     int region_id,
                                                     const std::string& context) {
    std::stringstream ss;
    ss << "material_" << model_type << "_region_" << region_id;

    if (!context.empty()) {
        ss << "_" << context;
    }

    ss << "_rank_" << mpi_rank << ".log";

    return ss.str();
}

void UnifiedLogger::print_capture_statistics() {
    // Gather statistics from all ranks
    int local_captures = stats_.total_captures;
    int total_captures = 0;

    // Use MPI to gather totals
    MPI_Reduce(&local_captures, &total_captures, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only rank 0 prints and scans directory
    if (mpi_rank == 0) {
        std::cout << "\n=== Material Output Summary ===" << std::endl;
        std::cout << "Total capture sessions: " << total_captures << std::endl;

        // Scan the log directory for all material files
        try {
            std::map<std::string, std::vector<std::filesystem::path>> files_by_type;
            int actual_file_count = 0;

            // Look for all files matching material log patterns
            for (const auto& entry : std::filesystem::directory_iterator(log_directory)) {
                if (!entry.is_regular_file())
                    continue;

                std::string filename = entry.path().filename().string();

                // Skip the main simulation log
                if (filename == main_log_filename.filename().string())
                    continue;

                // Check if it matches material log pattern
                if (filename.find("material_") == 0 && filename.find(".log") != std::string::npos) {
                    actual_file_count++;

                    // Extract model type from filename
                    std::string type = "other";
                    if (filename.find("ExaCMech") != std::string::npos ||
                        filename.find("exacmech") != std::string::npos) {
                        type = "ExaCMech";
                    } else if (filename.find("UMAT") != std::string::npos ||
                               filename.find("umat") != std::string::npos) {
                        type = "UMAT";
                    }
                    files_by_type[type].push_back(entry.path());
                } else if (filename.find("mfem") == 0 &&
                           filename.find(".log") != std::string::npos) {
                    actual_file_count++;
                    files_by_type["mfem"].push_back(entry.path());
                }
            }

            std::cout << "Files found in directory scan: " << actual_file_count << std::endl;

            if (actual_file_count > 0) {
                std::cout << "\nFiles with captured output:" << std::endl;

                // Print grouped files
                for (const auto& [type, files] : files_by_type) {
                    std::cout << "\n  " << type << " outputs (" << files.size()
                              << " files):" << std::endl;

                    // Group by region for cleaner output
                    std::map<int, std::vector<std::filesystem::path>> by_region;
                    for (const auto& file : files) {
                        std::string fname = file.filename().string();
                        // Extract region number
                        size_t region_pos = fname.find("region_");
                        if (region_pos != std::string::npos) {
                            int region = std::stoi(fname.substr(region_pos + 7));
                            by_region[region].push_back(file);
                        } else {
                            by_region[-1].push_back(file); // Unknown region
                        }
                    }

                    // Print by region
                    for (const auto& [region, region_files] : by_region) {
                        if (region >= 0) {
                            std::cout << "    Region " << region << ":" << std::endl;
                            for (const auto& file : region_files) {
                                // Get file size for additional info
                                auto size = std::filesystem::file_size(file);
                                std::cout << "      - " << file.filename().string() << " (" << size
                                          << " bytes)" << std::endl;
                            }
                        }
                    }

                    // Print files without clear region
                    if (by_region.count(-1) > 0) {
                        std::cout << "    MFEM:" << std::endl;
                        for (const auto& file : by_region[-1]) {
                            auto size = std::filesystem::file_size(file);
                            std::cout << "      - " << file.filename().string() << " (" << size
                                      << " bytes)" << std::endl;
                        }
                    }
                }
            }

        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Warning: Could not scan log directory: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// SHUTDOWN
// ============================================================================
void UnifiedLogger::shutdown() {
    // Step 1: End any active captures
    while (!capture_stack.empty()) {
        end_capture();
    }

    // Step 2: Print statistics
    print_capture_statistics();

    // Step 3: Write footer (rank 0 only)
    if (mpi_rank == 0) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "End Time: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                  << std::endl;
        std::cout << "Log files saved in: " << log_directory << std::endl;
    }

    // Step 4: Disable tee functionality
    disable_main_logging();

    // Step 5: Close main log file
    if (main_log_file) {
        main_log_file->close();
    }
}

// ============================================================================
// SCOPED CAPTURE IMPLEMENTATION
// ============================================================================
UnifiedLogger::ScopedCapture::ScopedCapture(const std::string& filename, bool suppress_non_zero)
    : logger(UnifiedLogger::get_instance()) {
    logger.begin_capture(filename, suppress_non_zero);
}

UnifiedLogger::ScopedCapture::~ScopedCapture() {
    // Save any created filename before end_capture
    filename_created = logger.end_capture();
}

} // namespace exaconstit