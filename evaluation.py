import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file_path', required=True)
parser.add_argument('-output_path', required=True)
parser.add_argument('-rev_output_path', default=None)
opt = parser.parse_args()

output = json.load(open(opt.file_path, 'r', encoding='utf-8'))
out_file = open(opt.output_path, 'w', encoding='utf-8')
if opt.rev_output_path:
    rev_out_file = open(opt.rev_output_path, 'w', encoding='utf-8')
for data in output:
    predict = data['prediction']
    out_file.write(predict + '\n')
    if opt.rev_output_path:
        rev_predict = data['rev_prediction']
        rev_out_file.write(rev_predict + '\n')

out_file.close()

