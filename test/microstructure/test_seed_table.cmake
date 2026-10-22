# Seed tables (ms_io: io/seed_table_io) and the ROI filler (ms_seeding: seeding/roi_filler,
# plus the exclusion box and id offset of seeding/seed_generators). The parse tests read the
# checked-in tables in test/data/microstructure/seed_tables; error cases are written at test time
# into the working directory (file names include the rank count, so the np variants never share a
# file). Every collective call is checked for rank-count independence by comparing with a
# single-rank reference computed in the same run. Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_seed_table
             SOURCES       test_seed_table.cpp
             DEPENDS_ON    ms_io ms_seeding ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

target_compile_definitions(
    test_seed_table PRIVATE
    MS_SEED_TABLE_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data/microstructure/seed_tables")
