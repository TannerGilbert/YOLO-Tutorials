import pandas as pd
import argparse
import os


def csv_to_yolo(input_filepath, output_path, image_folder):
	df = pd.read_csv(input_filepath)

	# get and save classes
	df['class_id'] = df['class'].map(lambda x: df['class'].unique().tolist().index(x))
	classes_file = open(os.path.join(output_path, 'classes.txt'), 'w')
	for c in df['class'].unique():
		classes_file.write(c + '\n')
	classes_file.close()	

	# create txt file
	output_file = open(os.path.join(output_path, 'train.txt'), 'w')
	for image_name in df['filename'].unique():
		image_path = os.path.join(image_folder,image_name).replace("\\", "/")
		output_file.write(f'{image_path} ')
		for index, row in df[(df['filename']==image_name)].iterrows():
			output_file.write(f'{row["xmin"]},{row["ymin"]},{row["xmax"]},{row["ymax"]},{row["class_id"]} ')
		output_file.write('\n')
	output_file.close()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="CSV to Yolo")
	parser.add_argument('-i', '--input', type=str, required=True, help='Path to csv file')
	parser.add_argument('-f', '--folder', type=str, default='', help='Image folder (if filenames are relativ)')
	parser.add_argument('-o', '--output', type=str, default='./', help='Output path')
	args = parser.parse_args()
	csv_to_yolo(args.input, args.output, args.folder)
