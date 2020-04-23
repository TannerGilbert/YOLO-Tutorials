import xml.etree.ElementTree as ET
import argparse
import os
import glob


def voc_to_yolo(input_path, output_path, output_filename, image_folder, classes):
    # save classes
    classes_file = open(os.path.join(output_path, 'classes.txt'), 'w')
    for c in classes:
        classes_file.write(c + '\n')
    classes_file.close()	

    # create txt file
    output_file = open(os.path.join(output_path, output_filename), 'w')
    for xml_file in glob.glob(input_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_path = os.path.join(image_folder,root.find("filename").text).replace("\\", "/")
        output_file.write(f'{image_path} ')
        for member in root.findall('object'):
            output_file.write(f'{int(member[4][0].text)},{int(member[4][1].text)},{int(member[4][2].text)},{int(member[4][3].text)},{classes.index(member[0].text)} ')
        output_file.write('\n')
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOC to YOLO')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input folder')
    parser.add_argument('-f', '--folder', type=str, default='', help='Image folder (if filenames are relativ)')
    parser.add_argument('-c', '--classes', nargs='+', required=True, help='Classes')
    parser.add_argument('-o', '--output', type=str, default='./', help='Output path')
    parser.add_argument('-of', '--output_filename', type=str, default='train.txt', help='Output Filename')
    args = parser.parse_args()
    voc_to_yolo(args.input, args.output, args.output_filename, args.folder, args.classes)