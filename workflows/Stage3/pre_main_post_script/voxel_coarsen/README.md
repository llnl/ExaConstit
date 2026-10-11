# Voxel Coarsen
This is a Rust crate with python bindings that performs the coarsening steps of the ExaCA data in preparation for an ExaConstit simulation.

This crate assumes the ExaCA file traverses data in the following axis order from fastest to slowest x->y->z (the order ExaCA currently writes its output in). If this changes then the logic in `rearrange_data` will need to be updated.

In order to keep compile times lowish, I make use of a data reader crate (an old personal project of mine) that functions in a similar manner to Python's loadtxt with fairly decent read / parsing speeds (150-250 MB/s on an old 2013 mac). This simple data reader is roughly 2x as slow as the Rust "Polars" crate which I have as an optional feature. I'm not using that optional feature by default as it has a compile on time on the 10s of minutes due to what appears to be the fat-LTO that LLVM is doing on everything.

In order to use the python bindings, it should be fairly simple. You'll need to make use of the `setuptools-rust` python tools. Therefore, you'll need to do:
`pip install setuptools-rust setuptools wheel`

and then you can build the crate and python bindings in one go by running the following command from the top level directory of `voxel_coarsen`:

`python setup.py develop --user`


On Crusher, you'll need to do the following first:
`ml PrgEnv-gnu/8.3.3`
`export CC=/opt/cray/pe/gcc/11.2.0/bin/gcc`
before you can run the above `python setup.py ...` command as this compiler is required to satisfy some build issues on Crusher.

Based on my understanding of things, this should create a user local version of things that should allow you to then just run the `exaconstit_preprocessing_main.py` script. If you run into any issues check out this page on how things should work: https://setuptools-rust.readthedocs.io/en/latest/README.html