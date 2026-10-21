// Placeholder translation unit for the ms_geometry library, which will hold the seed search
// structure (bounding volume hierarchies over the seeds); not implemented yet. It exists so the
// CMake target has at least one source file and links (ms_add_module fails at configure time on a
// module with no .cpp). It intentionally defines nothing.
#include "microstructure/microstructure_types.hpp"

namespace exaconstit::microstructure {} // namespace exaconstit::microstructure
