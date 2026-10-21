/**
 * @file output_writer.cpp
 * @brief Implementation of the microstructure output files (see output_writer.hpp for the file
 * formats and which ExaConstit reader consumes each one).
 *
 * Structure of this file:
 * - an anonymous namespace (a `namespace { ... }` block without a name, whose contents are visible
 *   only inside this .cpp file) with the shared helpers: opening a file and writing it plain or
 *   gzip-compressed (WriteStream / detail::WriteGzip), the collective error agreement
 *   (CollectiveTry), and input checks;
 * - the public writers, in the order of the header.
 *
 * Everything runs on the host. The only MPI communication is in WritePartitionedMesh,
 * WriteParallelMesh and WriteParaView (all collective).
 */

#include "microstructure/io/output_writer.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace exaconstit::microstructure {

namespace {

namespace fs = std::filesystem;

/**
 * Significant digits for every number written as text. 17 digits are enough for any IEEE double
 * to be read back to exactly the same bits, so coordinates and quaternions survive the round
 * trip through the file unchanged.
 */
constexpr int TEXT_PRECISION = 17;

/** Size in bytes of the uncompressed buffer zstr compresses at a time (1 MiB, zstr's default). */
constexpr std::size_t GZIP_BUFFER_BYTES = static_cast<std::size_t>(1) << 20;

/** True if @p s ends with @p suffix. */
bool EndsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * Creates the directory that will contain @p path (and its parents), if it does not exist yet.
 * Throws std::runtime_error with the operating system's reason if that fails.
 */
void CreateParentDirectories(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return; // bare file name: the current directory, which exists
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error("Cannot create directory " + parent.string() + ": " +
                                 ec.message());
    }
}

/** Throws std::invalid_argument unless 0 <= compression <= 9 (the zlib level range). */
void CheckCompressionLevel(int compression) {
    if (compression < 0 || compression > 9) {
        throw std::invalid_argument("Compression level must be in 0-9, got " +
                                    std::to_string(compression));
    }
}

#ifdef MFEM_USE_ZLIB
/**
 * A stream buffer that sits between zstr's compressor and the real file and hides write errors
 * from the compressor, while remembering them for us.
 *
 * A stream buffer (std::streambuf) is the object behind a C++ stream that actually moves the
 * bytes: std::ostream formats numbers into text and hands the characters to its stream buffer,
 * which stores or forwards them (a std::filebuf to a file, zstr::ostreambuf through zlib to
 * another stream buffer). Chaining buffers is how compression is slipped in between the
 * formatting and the file.
 *
 * Why it exists: zstr::ostreambuf (builds/mfem/general/zstr.hpp:496-605) flushes itself in its
 * destructor (the function C++ runs automatically when an object goes out of scope or is
 * deleted) by calling sync() (zstr.hpp:543-559). A destructor must not throw; if it does, C++
 * calls std::terminate and the whole MPI job dies without a message. zstr's sync() can throw a
 * zstr::Exception when zlib's deflate() returns an error (zstr.hpp:517-527), which zlib does when
 * the compressor's internal state is inconsistent. After a failed write to the sink, zstr leaves
 * that state half-finished: overflow() drops its buffer pointers mid-stream (zstr.hpp:566-569),
 * and sync() skips deflateReset() after an incomplete finish (zstr.hpp:589-593). With the
 * current zstr and zlib this still ends in a -1 return rather than a throw (checked with a
 * failing sink), but that depends on zlib internals we do not control.
 *
 * This buffer always tells zstr that every byte was written, so zstr never takes its error path,
 * its compressor always sees a normal, consistent stream, and the destructor's final sync()
 * cannot throw by construction. The real outcome is recorded in Failed(), which the caller
 * checks after the explicit flush. Once the real sink has failed, further bytes are dropped (the
 * file is unusable anyway).
 *
 * It has no buffer of its own (no setp() call), so every byte zstr sends arrives in xsputn() or
 * overflow() and is forwarded at once.
 */
class ErrorRecordingSink : public std::streambuf {
public:
    explicit ErrorRecordingSink(std::streambuf& target) : m_target(target) {}

    /** True once any forward to the real sink wrote fewer bytes than asked, or threw. */
    bool Failed() const { return m_failed; }

protected:
    /** Forwards @p count bytes; reports all of them written regardless of the outcome. */
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (!m_failed) {
            std::streamsize written = -1;
            try {
                written = m_target.sputn(data, count);
            } catch (...) {
                written = -1; // a throwing sink counts as a failed write
            }
            if (written != count) {
                m_failed = true;
            }
        }
        return count;
    }

    /** Single-character path (std::streambuf calls it when there is no put buffer). */
    int_type overflow(int_type c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            const char ch = traits_type::to_char_type(c);
            xsputn(&ch, 1);
        }
        return traits_type::not_eof(c);
    }

    /**
     * Flush request: passed on to the real sink, failure recorded, success always reported.
     *
     * In practice this is never called: zstr::ostreambuf::sync() finishes the gzip data with
     * sputn() calls only and never calls pubsync() on its sink (zstr.hpp:578-594). It is
     * implemented so the class is a correct stream buffer on its own. Bytes still sitting in the
     * file's own buffer (std::filebuf) are flushed by file.close() in WriteStream, and a
     * disk-full error at that point is reported there.
     */
    int sync() override {
        if (!m_failed) {
            try {
                if (m_target.pubsync() != 0) {
                    m_failed = true;
                }
            } catch (...) {
                m_failed = true;
            }
        }
        return 0;
    }

private:
    std::streambuf& m_target;
    bool m_failed = false;
};
#endif

/**
 * Opens @p path for writing, lets @p write produce the content on a std::ostream with precision
 * 17, closes the file, and throws std::runtime_error if anything went wrong.
 *
 * With @p compression > 0 the content is gzip-compressed at that level through
 * detail::WriteGzip. mfem::ofgzstream is not used for that because it ignores the requested
 * level: its bool constructor always uses the zlib default and its mode-string constructor
 * always 6 (zstr.hpp:717-765). Callers must have reduced @p compression to 0 already when MFEM
 * has no zlib (EffectiveCompression).
 *
 * Steps: create parent directories; open the file in binary mode (so no newline translation
 * corrupts gzip bytes on any platform); write plain or compressed; close and check the stream
 * state. close() flushes the file buffer, so a full disk is reported here and not lost.
 */
template <typename WriteFn>
void WriteStream(const fs::path& path, int compression, WriteFn&& write) {
    CreateParentDirectories(path);
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string() + " for writing");
    }

    if (compression > 0) {
        // Compressed: the gzip writer pushes its output into the file's stream buffer
        detail::WriteGzip(*file.rdbuf(), compression, std::forward<WriteFn>(write),
                          path.string());
    } else {
        // Plain text: write straight into the file stream
        file.precision(TEXT_PRECISION);
        write(file);
    }

    // close() flushes what is still buffered; failbit is set if that or any earlier write failed
    file.close();
    if (file.fail()) {
        throw std::runtime_error("Error writing " + path.string());
    }
}

/**
 * The compression level that will actually be used: the requested level when MFEM has zlib,
 * otherwise 0. Validates the range first.
 *
 * @param warn_if_unavailable Print a warning when compression was asked for but MFEM has no
 *        zlib. Collective writers pass true only on rank 0, so the warning appears once rather
 *        than once per rank.
 */
int EffectiveCompression(int compression, bool warn_if_unavailable) {
    CheckCompressionLevel(compression);
#ifndef MFEM_USE_ZLIB
    if (compression > 0 && warn_if_unavailable) {
        MFEM_WARNING("MFEM was built without zlib; writing uncompressed mesh files");
    }
    return 0;
#else
    (void)warn_if_unavailable;
    return compression;
#endif
}

/**
 * Turns free-form provenance text into MFEM mesh comment lines: every line that does not start
 * with '#' gets a "# " prefix, so MFEM's mesh reader skips it. Lines are joined with '\n', with
 * no trailing newline (Mesh::Print adds its own).
 */
std::string FormatComments(const std::string& comments) {
    if (comments.empty()) {
        return comments;
    }
    std::istringstream in(comments);
    std::ostringstream out;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (!first) {
            out << '\n';
        }
        first = false;
        if (line.empty() || line[0] != '#') {
            out << "# ";
        }
        out << line;
    }
    return out.str();
}

/**
 * Runs @p local_work on this rank, then makes every rank of @p comm agree on whether it failed
 * anywhere, and if so throws std::runtime_error on every rank. Collective.
 *
 * Why: if rank 2 fails to open a file and throws while the other ranks carry on into the next
 * MPI call (a broadcast, a barrier), those ranks wait forever for rank 2 and the job hangs
 * instead of stopping with an error. Here each rank catches its own exception, all ranks combine
 * a 0/1 failure flag with MPI_Allreduce(MAX) (one number in, the largest over all ranks out on
 * every rank), and then all ranks throw together. The rank that failed reports its own message;
 * the others report that another rank failed.
 */
template <typename Fn>
void CollectiveTry(MPI_Comm comm, const std::string& what, Fn&& local_work) {
    // Step 1: local work, with any exception turned into a flag and a message
    int failed = 0;
    std::string message;
    try {
        local_work();
    } catch (const std::exception& e) {
        failed = 1;
        message = e.what();
    }

    // Step 2: agree on failure over all ranks
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);

    // Step 3: throw on every rank together
    if (any_failed) {
        if (failed) {
            throw std::runtime_error(what + ": " + message);
        }
        throw std::runtime_error(what + ": failed on another rank");
    }
}

/** Throws std::invalid_argument unless an optional grain-table column is empty or has
 * @p expected values. */
void CheckColumn(std::size_t size, std::size_t expected, const char* name) {
    if (size != 0 && size != expected) {
        throw std::invalid_argument(std::string("Grain table column ") + name + " has " +
                                    std::to_string(size) + " values, expected " +
                                    std::to_string(expected));
    }
}

} // namespace

/**
 * Gzip writer (declared in output_writer.hpp, namespace detail).
 *
 * The stream chain is: our std::ostream -> zstr::ostreambuf (compresses, zstr.hpp:496) ->
 * ErrorRecordingSink (hides and records write errors) -> @p sink (normally the file).
 *
 * Steps:
 * 1. Build the chain and let @p write produce the text.
 * 2. Flush explicitly. std::ostream::flush calls zstr's sync(), which compresses what is buffered
 *    and finishes a complete gzip member (zstr.hpp:578-594). flush() catches any exception from
 *    the buffer and sets badbit instead (the C++ standard's rules for unformatted output), so
 *    step 2 never throws.
 * 3. Leave the scope: zstr's destructor calls sync() once more, which only appends an empty
 *    gzip member (about 20 bytes; every gzip reader, including zstr::istreambuf and `gzip -d`,
 *    reads concatenated members as one stream). Because the sink never reports a failure to
 *    zstr, its compressor is in a consistent state and this cannot throw (see
 *    ErrorRecordingSink).
 * 4. Check the stream state and the recorded sink failure, and throw std::runtime_error.
 *
 * If @p write itself throws, the exception propagates and step 3 runs during stack unwinding
 * (C++ leaving each enclosing scope on the way out of a throw, running the destructors of the
 * objects in them), which is safe for the same reason.
 */
void detail::WriteGzip(std::streambuf& sink,
                       int level,
                       const std::function<void(std::ostream&)>& write,
                       const std::string& what) {
    if (level < 1 || level > 9) {
        throw std::invalid_argument("WriteGzip: gzip level must be in 1-9, got " +
                                    std::to_string(level));
    }
#ifdef MFEM_USE_ZLIB
    ErrorRecordingSink guarded_sink(sink);
    bool stream_ok = false;
    {
        // Step 1: the compressing chain. The zstr constructor throws zstr::Exception (a
        // std::exception) only if zlib cannot initialize, e.g. out of memory.
        zstr::ostreambuf zbuf(&guarded_sink, GZIP_BUFFER_BYTES, level);
        std::ostream os(&zbuf);
        os.precision(TEXT_PRECISION);
        write(os);

        // Step 2: finish the gzip data now, so errors are seen here and not in the destructor
        os.flush();
        stream_ok = static_cast<bool>(os);
    } // Step 3: zbuf's destructor runs its final (non-throwing) sync here

    // Step 4: report what went wrong
    if (!stream_ok || guarded_sink.Failed()) {
        throw std::runtime_error("Error writing compressed data to " + what);
    }
#else
    (void)sink;
    (void)write;
    throw std::runtime_error("Cannot write " + what + " compressed: MFEM was built without zlib");
#endif
}

/** Copies the per-grain columns; the remaining columns stay empty for the caller. */
GrainTable MakeGrainTable(const MicrostructureResult& result) {
    GrainTable table;
    table.num_grains = result.num_grains;
    table.quat = result.grain_quat;
    table.region = result.grain_region;
    table.source_id = result.grain_source_id;
    return table;
}

/**
 * Serial mesh writer.
 *
 * Steps: reject a ParMesh; settle the compression level; fix the ".gz" suffix so the name says
 * what the file is; write with Mesh::Print at 17 digits (Mesh::Print sets no precision itself,
 * so the stream's precision is what is used).
 *
 * Mesh::Print(std::ostream&, const std::string& comments) (mesh.hpp:2578) puts the comments
 * after the first line ("MFEM mesh v1.0"), followed by a blank line.
 */
std::filesystem::path WriteSerialMesh(const mfem::Mesh& serial_with_attributes,
                                      const std::filesystem::path& path,
                                      int compression,
                                      const std::string& comments) {
    // A ParMesh is also an mfem::Mesh, so check the dynamic type: the actual class of the object
    // behind the reference, which may be a class derived from the declared type mfem::Mesh.
    // dynamic_cast returns nullptr when the object is not a ParMesh.
    if (dynamic_cast<const mfem::ParMesh*>(&serial_with_attributes) != nullptr) {
        throw std::invalid_argument("WriteSerialMesh needs a serial mesh, got a ParMesh");
    }
    // Not collective: whichever rank calls this warns (if MFEM has no zlib)
    const int level = EffectiveCompression(compression, true);

    // Name: append ".gz" when compressing, drop it when not
    std::string name = path.string();
    if (level > 0) {
        if (!EndsWith(name, ".gz")) {
            name += ".gz";
        }
    } else if (EndsWith(name, ".gz")) {
        name.erase(name.size() - 3);
    }
    const fs::path out_path(name);
    const std::string formatted = FormatComments(comments);

    WriteStream(out_path, level, [&](std::ostream& os) {
        // Qualified call Mesh::Print: always the serial printer, even if a derived class
        // overrides Print
        serial_with_attributes.Mesh::Print(os, formatted);
    });
    return out_path;
}

/**
 * Partitioned (parallel-format) mesh writer.
 *
 * Algorithm:
 * 1. Validate the arguments identically on every rank (every rank has the same mesh and
 *    arguments, so either all ranks throw here or none, before any communication).
 * 2. Rank 0 computes the element-to-part map with METIS, the graph-partitioning library MFEM
 *    uses to split a mesh into balanced, compact pieces (Mesh::GeneratePartitioning, which
 *    returns memory allocated with new[] that we must delete[], mesh.cpp:9010-9032) and checks
 *    that no part is empty, since an empty part would give a rank with no elements. The ranks
 *    agree on the outcome (CollectiveTry), then rank 0 broadcasts the map. METIS should be
 *    deterministic anyway, but one broadcast map guarantees every rank cuts the same parts.
 * 3. Each rank with rank < num_parts builds an mfem::MeshPartitioner (mesh.hpp:3003-3040) and
 *    handles the parts p = rank, rank + nranks, rank + 2 nranks, ... (round-robin): ExtractPart
 *    copies part p's elements, boundary, vertices (and Nodes for a curved mesh), plus the shared
 *    vertices/edges/faces with the other parts, into an mfem::MeshPart; MeshPart::Print
 *    (mesh.cpp:14348) writes it as "MFEM mesh v1.2" with the communication groups ParMesh needs.
 *    A final CollectiveTry makes all ranks agree whether every part was written.
 */
std::vector<std::filesystem::path> WritePartitionedMesh(mfem::Mesh& serial_with_attributes,
                                                        int num_parts,
                                                        const std::string& prefix,
                                                        int compression,
                                                        MPI_Comm comm) {
    // Step 1: argument checks (identical on every rank, no communication yet)
    if (dynamic_cast<const mfem::ParMesh*>(&serial_with_attributes) != nullptr) {
        throw std::invalid_argument("WritePartitionedMesh needs a serial mesh, got a ParMesh");
    }
    const int num_elements = serial_with_attributes.GetNE();
    if (num_parts < 1 || num_parts > num_elements) {
        throw std::invalid_argument("WritePartitionedMesh: num_parts = " +
                                    std::to_string(num_parts) + " must be in 1.." +
                                    std::to_string(num_elements));
    }
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    // Same level on every rank; only rank 0 prints the no-zlib warning
    const int level = EffectiveCompression(compression, rank == 0);

    // Part file names: prefix + part number zero-padded to 6 digits, no ".gz" (see header)
    std::vector<fs::path> names(static_cast<std::size_t>(num_parts));
    for (int p = 0; p < num_parts; ++p) {
        names[static_cast<std::size_t>(p)] = mfem::MakeParFilename(prefix, p);
    }

    // Step 2: rank 0 partitions and checks for empty parts; everyone agrees, then broadcast
    mfem::Array<int> partitioning(num_elements); // partitioning[e] = part of element e, 0-based
    CollectiveTry(comm, "WritePartitionedMesh", [&]() {
        if (rank != 0) {
            return;
        }
        CreateParentDirectories(names[0]);
        int* part = serial_with_attributes.GeneratePartitioning(num_parts, 1);
        std::copy(part, part + num_elements, partitioning.GetData());
        delete[] part;

        // Count elements per part; any empty part is an error
        std::vector<int> part_size(static_cast<std::size_t>(num_parts), 0);
        for (int e = 0; e < num_elements; ++e) {
            ++part_size[static_cast<std::size_t>(partitioning[e])];
        }
        for (int p = 0; p < num_parts; ++p) {
            if (part_size[static_cast<std::size_t>(p)] == 0) {
                throw std::runtime_error("the partitioner left part " + std::to_string(p) +
                                         " empty; use fewer parts");
            }
        }
    });
    MPI_Bcast(partitioning.GetData(), num_elements, MPI_INT, 0, comm);

    // Step 3: round-robin extraction and writing, then agreement on success
    CollectiveTry(comm, "WritePartitionedMesh", [&]() {
        if (rank >= num_parts) {
            return; // more ranks than parts: this rank has nothing to write
        }
        mfem::MeshPartitioner partitioner(serial_with_attributes, num_parts,
                                          partitioning.GetData());
        mfem::MeshPart mesh_part; // reused for each part this rank writes
        for (int p = rank; p < num_parts; p += nranks) {
            partitioner.ExtractPart(p, mesh_part);
            WriteStream(names[static_cast<std::size_t>(p)], level,
                        [&](std::ostream& os) { mesh_part.Print(os); });
        }
    });
    return names;
}

/**
 * ParMesh writer: every rank writes its own piece with ParMesh::ParPrint (pmesh.hpp:721), which
 * writes the same "MFEM mesh v1.2" parallel format as MeshPart::Print (local elements plus the
 * shared entities), so ParMesh(comm, istream) rebuilds the identical decomposition.
 *
 * Steps:
 * 1. Reject NURBS and nonconforming meshes. For those ParPrint does not write the v1.2 format
 *    (pmesh.cpp:6353-6364): a NURBS mesh falls back to a serial v1.0 Print of the local piece,
 *    without the shared-entity data readParallelMesh needs, and a nonconforming mesh is written
 *    in MFEM's NC format, which the microstructure pipeline does not support. Whether the mesh
 *    is NURBS (NURBSext set) or nonconforming is a property of the whole ParMesh, the same on
 *    every rank, so every rank throws here together without any communication.
 * 2. Rank 0 creates the output directory and all ranks agree it worked (otherwise ranks other
 *    than 0 could try to open files in a directory that does not exist yet).
 * 3. Each rank writes its part and all ranks agree on success.
 */
std::vector<std::filesystem::path> WriteParallelMesh(const mfem::ParMesh& pmesh,
                                                     const std::string& prefix,
                                                     int compression,
                                                     const std::string& comments) {
    const MPI_Comm comm = pmesh.GetComm();
    const int rank = pmesh.GetMyRank();
    const int nranks = pmesh.GetNRanks();
    const int level = EffectiveCompression(compression, rank == 0);

    // Step 1: only conforming, non-NURBS meshes have a parallel v1.2 ParPrint
    if (pmesh.NURBSext != nullptr) {
        throw std::invalid_argument("WriteParallelMesh: NURBS meshes are not supported "
                                    "(ParMesh::ParPrint writes them as serial v1.0 pieces that "
                                    "cannot be read back as a decomposition)");
    }
    if (pmesh.Nonconforming()) {
        throw std::invalid_argument("WriteParallelMesh: nonconforming meshes are not supported "
                                    "(ParMesh::ParPrint writes them in the NC mesh format)");
    }

    // Names of every rank's part, so all ranks can return the full list
    std::vector<fs::path> names(static_cast<std::size_t>(nranks));
    for (int p = 0; p < nranks; ++p) {
        names[static_cast<std::size_t>(p)] = mfem::MakeParFilename(prefix, p);
    }

    // Step 2: the directory must exist before any rank opens its file
    CollectiveTry(comm, "WriteParallelMesh", [&]() {
        if (rank == 0) {
            CreateParentDirectories(names[0]);
        }
    });

    // Step 3: each rank writes its own piece
    const std::string formatted = FormatComments(comments);
    CollectiveTry(comm, "WriteParallelMesh", [&]() {
        WriteStream(names[static_cast<std::size_t>(rank)], level,
                    [&](std::ostream& os) { pmesh.ParPrint(os, formatted); });
    });
    return names;
}

/** Orientation file: quaternion g on line g, values separated by single spaces. */
void WriteOrientationFile(const std::filesystem::path& path,
                          const std::vector<double>& grain_quat) {
    if (grain_quat.size() % 4 != 0) {
        throw std::invalid_argument("WriteOrientationFile: grain_quat size " +
                                    std::to_string(grain_quat.size()) +
                                    " is not a multiple of 4");
    }
    WriteStream(path, 0, [&](std::ostream& os) {
        // i is the index of the w component of one grain's quaternion
        for (std::size_t i = 0; i < grain_quat.size(); i += 4) {
            os << grain_quat[i] << ' ' << grain_quat[i + 1] << ' ' << grain_quat[i + 2] << ' '
               << grain_quat[i + 3] << '\n';
        }
    });
}

/** Region map: line g holds "g region_of_g", g 1-based. */
void WriteRegionMap(const std::filesystem::path& path, const std::vector<int>& grain_region) {
    WriteStream(path, 0, [&](std::ostream& os) {
        for (std::size_t g = 0; g < grain_region.size(); ++g) {
            os << (g + 1) << ' ' << grain_region[g] << '\n';
        }
    });
}

/** Grain file: line e+1 holds the grain id of element e (serial element order). */
void WriteGrainFile(const std::filesystem::path& path, const int* serial_grain_ids, std::size_t n) {
    if (n > 0 && serial_grain_ids == nullptr) {
        throw std::invalid_argument("WriteGrainFile: null grain id array");
    }
    WriteStream(path, 0, [&](std::ostream& os) {
        for (std::size_t e = 0; e < n; ++e) {
            os << serial_grain_ids[e] << '\n';
        }
    });
}

void WriteGrainFile(const std::filesystem::path& path, const std::vector<int>& serial_grain_ids) {
    WriteGrainFile(path, serial_grain_ids.data(), serial_grain_ids.size());
}

void WriteGrainFile(const std::filesystem::path& path,
                    const mfem::Array<int>& serial_grain_ids) {
    // HostRead makes the host copy valid (copying from the device if needed) and returns it
    WriteGrainFile(path, serial_grain_ids.HostRead(),
                   static_cast<std::size_t>(serial_grain_ids.Size()));
}

/**
 * Grain table CSV writer: check every column length first (so a bad table writes nothing),
 * then a header row naming the present columns, then one row per grain in the same column order.
 */
void WriteGrainTable(const std::filesystem::path& path, const GrainTable& table) {
    // Column length checks
    if (table.num_grains < 0) {
        throw std::invalid_argument("WriteGrainTable: negative num_grains");
    }
    const auto n = static_cast<std::size_t>(table.num_grains);
    if (table.quat.size() != 4 * n) {
        throw std::invalid_argument("Grain table column quat has " +
                                    std::to_string(table.quat.size()) + " values, expected " +
                                    std::to_string(4 * n));
    }
    CheckColumn(table.region.size(), n, "region");
    CheckColumn(table.source_id.size(), n, "source_id");
    CheckColumn(table.num_elements.size(), n, "num_elements");
    CheckColumn(table.volume.size(), n, "volume");
    CheckColumn(table.centroid.size(), 3 * n, "centroid");

    WriteStream(path, 0, [&](std::ostream& os) {
        // Header row
        os << "grain_id,qw,qx,qy,qz";
        if (!table.region.empty()) {
            os << ",region";
        }
        if (!table.source_id.empty()) {
            os << ",source_id";
        }
        if (!table.num_elements.empty()) {
            os << ",num_elements";
        }
        if (!table.volume.empty()) {
            os << ",volume";
        }
        if (!table.centroid.empty()) {
            os << ",cx,cy,cz";
        }
        os << '\n';

        // One row per grain; g is 0-based, the printed grain id g + 1
        for (std::size_t g = 0; g < n; ++g) {
            os << (g + 1);
            for (std::size_t k = 0; k < 4; ++k) {
                os << ',' << table.quat[4 * g + k];
            }
            if (!table.region.empty()) {
                os << ',' << table.region[g];
            }
            if (!table.source_id.empty()) {
                os << ',' << table.source_id[g];
            }
            if (!table.num_elements.empty()) {
                os << ',' << table.num_elements[g];
            }
            if (!table.volume.empty()) {
                os << ',' << table.volume[g];
            }
            if (!table.centroid.empty()) {
                os << ',' << table.centroid[3 * g] << ',' << table.centroid[3 * g + 1] << ','
                   << table.centroid[3 * g + 2];
            }
            os << '\n';
        }
    });
}

/**
 * ParaView writer.
 *
 * Steps:
 * 1. Every rank checks its own input; all ranks agree (CollectiveTry) before the collective
 *    ParFiniteElementSpace construction, so bad input on one rank cannot hang the others.
 * 2. Build an order-0 L2 space: exactly one degree of freedom per element, so the field is the
 *    element value itself. Fill the grain (and region) field element by element.
 * 3. Save with mfem::ParaViewDataCollection (each rank writes its own .vtu piece, rank 0 the
 *    .pvd/.pvtu index), then agree on the collection's error state. DataCollection::Error()
 *    is compared with No_Error (datacollection.hpp:380-392; NO_ERROR may be a system macro).
 */
void WriteParaView(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& element_grain,
                   const std::vector<int>& grain_region,
                   const std::filesystem::path& path) {
    const int ne = pmesh.GetNE();

    // Step 1: local input checks, agreed over all ranks
    CollectiveTry(pmesh.GetComm(), "WriteParaView", [&]() {
        if (element_grain.Size() != ne) {
            throw std::invalid_argument("element_grain has " +
                                        std::to_string(element_grain.Size()) +
                                        " entries for " + std::to_string(ne) + " elements");
        }
        if (grain_region.empty()) {
            return; // no region field, nothing more to check
        }
        const int* grains = element_grain.HostRead();
        for (int e = 0; e < ne; ++e) {
            const bool has_region =
                grains[e] >= 1 && static_cast<std::size_t>(grains[e]) <= grain_region.size();
            if (!has_region) {
                throw std::invalid_argument("grain id " + std::to_string(grains[e]) +
                                            " has no region");
            }
        }
    });

    // Step 2: piecewise-constant fields, one value per element
    mfem::L2_FECollection fec(0, pmesh.Dimension());
    mfem::ParFiniteElementSpace fes(&pmesh, &fec);
    mfem::ParGridFunction grain_gf(&fes);
    mfem::ParGridFunction region_gf(&fes);
    const int* grains = element_grain.HostRead();
    mfem::Array<int> dofs;
    for (int e = 0; e < ne; ++e) {
        fes.GetElementDofs(e, dofs); // a single dof for an order-0 L2 element
        grain_gf(dofs[0]) = grains[e];
        if (!grain_region.empty()) {
            region_gf(dofs[0]) = grain_region[static_cast<std::size_t>(grains[e] - 1)];
        }
    }

    // Step 3: write the collection and agree on its error state
    const fs::path prefix = path.parent_path();
    mfem::ParaViewDataCollection dc(path.filename().string(), &pmesh);
    if (!prefix.empty()) {
        dc.SetPrefixPath(prefix.string());
    }
    dc.RegisterField("grain_id", &grain_gf);
    if (!grain_region.empty()) {
        dc.RegisterField("region_id", &region_gf);
    }
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();
    CollectiveTry(pmesh.GetComm(), "WriteParaView", [&]() {
        if (dc.Error() != mfem::DataCollection::No_Error) {
            throw std::runtime_error("writing " + path.string() + " failed");
        }
    });
}

} // namespace exaconstit::microstructure
