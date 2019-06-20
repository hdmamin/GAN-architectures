import argparse
import ast
import matplotlib
matplotlib.use('Agg')

# Must import this after setting display to Agg.
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys


def get_args():
    """Create parser to get user-specified params."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, 
                        help='name of doodle category (e.g. dog)')
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='start index - only need a nonzero value when'
                        'resuming image generation')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='end index - use to test on a subset of images')
    
    return parser.parse_args()
    
    
def coords2img(row, out_file, out_dir):
    """Convert the line coordinates, stored as a string in the raw dataset,
    into an image and save it in the output directory.
    
    Parameters
    -----------
    row: str
        One item in our df/series corresponding to a single image.
    out_file: str
        File name of output.
    out_dir: str
        Name of output directory.
    """
    # For convenient dataloader usage, nest image dir inside another dir.
    nested_dir = os.path.join(out_dir, f'{out_dir}_data')
    output_file = os.path.join(nested_dir, str(out_file))
    if not os.path.exists(nested_dir):
        os.makedirs(nested_dir)
    
    lines = [np.array(line)/255 for line in ast.literal_eval(row)]
    # Save as 64px x 64px img. Flip y axis because 0,0 is given as top left.
    fig, ax = plt.subplots(figsize=(.64, .64), dpi=100)
    plt.gca().invert_yaxis()
    for i, line in enumerate(lines):
        ax.plot(line[0, :], line[1, :], '-', c='black')
    plt.axis('off')
    plt.savefig(output_file, dpi=100)
    plt.close()
    

def gen_images(data, out_dir, start=0, end=None):
    """Generate images for all examples in our series.
    
    Parameters
    -----------
    data: pd.Series
        Contains stroke coordinatess for all images.
    start: int
        Index of file to start with (useful if memory error forces us to 
        resume in the middle - leave as zero to start from the beginning).
    end: int
        Last file to create (useful for testing - leave as None to create all 
        samples).
    """
    for i, row in enumerate(data[start:end], start):
        if i % 1000 == 0: print(f'Processing file {i}...')
        coords2img(row, i, out_dir)
        
    
def main():
    args = get_args()
    df = pd.read_csv(args.file, usecols=[1, 3])
    n_total = df.shape[0]
    df = df.loc[df.recognized, 'drawing']
    n_recognized = df.shape[0]
    print(f'Generating {n_recognized} samples '
          f'({n_total - n_recognized} dropped)...')
    
    out_dir = args.file.split('.')[0]
    gen_images(df, out_dir, args.start, args.end)
    
    
if __name__ == '__main__':
    main()
                        