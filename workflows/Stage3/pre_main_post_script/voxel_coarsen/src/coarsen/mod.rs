use anyhow::{format_err, Error};
use rand::{thread_rng, Rng};

#[cfg(not(feature = "polars"))]
use data_reader::reader;
#[cfg(feature = "polars")]
use polars::prelude::*;

use std::cmp::Ordering;
use std::convert::TryFrom;

#[cfg(not(feature = "polars"))]
fn read_data(f: &str) -> Result<Box<dyn reader::ReaderResults<i32>>, Error> {
    let params = reader::ReaderParams {
        comments: Some(b'%'),
        delimiter: reader::Delimiter::Any(b','),
        skip_header: Some(2),
        skip_footer: None,
        usecols: None,
        max_rows: None,
        row_format: false,
        ..Default::default()
    };

    reader::load_txt_i32(f, &params)
}

#[cfg(feature = "polars")]
fn read_data(f: &str) -> PolarsResult<DataFrame> {
    // Same layout as the data_reader version: 2 header lines to skip, comma
    // separated columns, and '%' starting a comment.
    CsvReadOptions::default()
        .with_has_header(false)
        .with_skip_rows(2)
        .map_parse_options(|opts| opts.with_separator(b',').with_comment_prefix(Some("%")))
        .try_into_reader_with_file_path(Some(f.into()))?
        .finish()
}

fn sort_data(chunk_size: usize, data: &mut [i32]) {
    assert!(data.len() % chunk_size.pow(3) == 0);
    let c3 = chunk_size.pow(3);
    data.chunks_exact_mut(c3).for_each(|chunk| {
        chunk.sort();
    });
}

fn rearrange_data(
    orig_data: &[i32],
    box_size: &(usize, usize, usize),
    chunk_size: usize,
    data: &mut [i32],
) {
    assert!(orig_data.len() >= (box_size.0 * box_size.1 * box_size.2));
    assert!(data.len() >= (box_size.0 * box_size.1 * box_size.2));
    assert!(chunk_size.pow(3) <= (box_size.0 * box_size.1 * box_size.2));

    orig_data.iter().enumerate().for_each(|(index, val)| {
        // CA results are indexed as x being fastest axis indexed on,
        // then y axis, and finally the z axis.
        let i = index % box_size.0;
        let j = (index / (box_size.0)) % box_size.1;
        let k = index / (box_size.0 * box_size.1);
        // We want our outputted data to be re-ordered such that
        // x is the fastest indexed item, then y axis, and finally the z axis.
        let block = (i / chunk_size)
            + ((j / chunk_size) * (box_size.0 / chunk_size))
            + ((k / chunk_size) * ((box_size.0 * box_size.1) / chunk_size.pow(2)));
        let offset = block * (chunk_size.pow(3))
            + (i % chunk_size)
            + (j % chunk_size * chunk_size)
            + (k % chunk_size * (chunk_size.pow(2)));
        data[offset] = *val;
    });
}

fn coarsen(orig_data: &[i32], block_size: usize, coarse_data: &mut [i32]) {
    const MAX_SIZE: usize = 32;
    let mut rng = thread_rng();

    coarse_data
        .iter_mut()
        .zip(orig_data.chunks_exact(block_size))
        .for_each(|(coarse_val, orig_chunk)| {
            let mut index_arr = 0;
            let mut prev_val = i32::MIN;
            let mut count = 0;
            let mut max_count = 0;
            let mut max_array = [0i32; MAX_SIZE];

            orig_chunk.iter().for_each(|val| {
                if prev_val != *val {
                    prev_val = *val;
                    count = 1;
                } else {
                    count += 1;
                }

                match count.cmp(&max_count) {
                    Ordering::Greater => {
                        index_arr = 0;
                        max_array[0] = *val;
                        max_count = count;
                    }
                    Ordering::Equal => {
                        index_arr += 1;
                        if index_arr == MAX_SIZE {
                            index_arr = MAX_SIZE - 1;
                        }
                        max_array[index_arr] = *val;
                    }
                    Ordering::Less => (),
                }
            });

            // Will use this to get out a random value if we obtain equal
            let ind = rng.gen_range(0..(index_arr + 1));
            *coarse_val = max_array[ind];
        });
}

#[cfg(not(feature = "polars"))]
pub fn voxel_coarsen(file: &str, coarsen_size: usize) -> Result<((usize, usize, usize), Vec<i32>), Error> {
    let read_results = read_data(file)?;

    // Find the box size of things
    let box_size = {
        let x = read_results.get_inner_lanes(0);
        let y = read_results.get_inner_lanes(1);
        let z = read_results.get_inner_lanes(2);
        let mut min = (0, 0, 0);
        let mut max = (0, 0, 0);
        min.0 = *x.iter().min().unwrap();
        max.0 = *x.iter().max().unwrap();
        min.1 = *y.iter().min().unwrap();
        max.1 = *y.iter().max().unwrap();
        min.2 = *z.iter().min().unwrap();
        max.2 = *z.iter().max().unwrap();
        (
            usize::try_from(max.0 - min.0 + 1).unwrap(),
            usize::try_from(max.1 - min.1 + 1).unwrap(),
            usize::try_from(max.2 - min.2 + 1).unwrap(),
        )
    };

    if (box_size.0 % coarsen_size != 0)
        || (box_size.1 % coarsen_size != 0)
        || (box_size.2 % coarsen_size != 0)
    {
        return Err(format_err!(
            "CA box size {:?} is not divisible by input coarsen size {}",
            box_size,
            coarsen_size
        ));
    }

    let mut data = read_results.get_col(3);
    let col_result = read_results.get_inner_lanes(3);
    rearrange_data(col_result, &box_size, coarsen_size, &mut data);

    sort_data(coarsen_size, &mut data);

    let block_size = coarsen_size.pow(3);
    let mut coarse_data: Vec<i32> = vec![0i32; data.len() / block_size];

    coarsen(&data, block_size, &mut coarse_data);

    Ok((box_size, coarse_data))
}

#[cfg(feature = "polars")]
pub fn voxel_coarsen(file: &str, coarsen_size: usize) -> Result<((usize, usize, usize), Vec<i32>), Error> {
    let df = read_data(file)?;
    let box_size = {
        let x = df[0].i64()?;
        let y = df[1].i64()?;
        let z = df[2].i64()?;
        let mut min = (0, 0, 0);
        let mut max = (0, 0, 0);
        min.0 = x.into_no_null_iter().min().unwrap();
        max.0 = x.into_no_null_iter().max().unwrap();
        min.1 = y.into_no_null_iter().min().unwrap();
        max.1 = y.into_no_null_iter().max().unwrap();
        min.2 = z.into_no_null_iter().min().unwrap();
        max.2 = z.into_no_null_iter().max().unwrap();
        (
            usize::try_from(max.0 - min.0 + 1).unwrap(),
            usize::try_from(max.1 - min.1 + 1).unwrap(),
            usize::try_from(max.2 - min.2 + 1).unwrap(),
        )
    };

    if (box_size.0 % coarsen_size != 0)
        || (box_size.1 % coarsen_size != 0)
        || (box_size.2 % coarsen_size != 0)
    {
        return Err(format_err!(
            "CA box size {:?} is not divisible by input coarsen size {}",
            box_size,
            coarsen_size
        ));
    }

    let mut data: Vec<i32> = df[3]
        .i64()?
        .into_no_null_iter()
        .map(|x| i32::try_from(x).unwrap())
        .collect();

    rearrange_data(&data.clone(), &box_size, coarsen_size, &mut data);
    sort_data(coarsen_size, &mut data);

    let block_size = coarsen_size.pow(3);
    let mut coarse_data: Vec<i32> = vec![0i32; data.len() / block_size];

    coarsen(&data, block_size, &mut coarse_data);

    Ok((box_size, coarse_data))
}
