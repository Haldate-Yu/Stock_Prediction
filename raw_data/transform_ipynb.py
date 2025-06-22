import json
import sys


def convert_ipynb_to_py(ipynb_file, py_file):
    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        with open(py_file, 'w', encoding='utf-8') as f:
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    # 写入代码前添加注释分隔符
                    f.write(f"# {'-' * 70}\n")
                    f.write(f"# Cell\n")
                    f.write(f"# {'-' * 70}\n\n")

                    # 写入代码内容
                    for line in cell['source']:
                        f.write(line)
                    f.write('\n\n')

        print(f"成功将 {ipynb_file} 转换为 {py_file}")
    except Exception as e:
        print(f"转换失败: {e}")


if __name__ == "__main__":
    convert_ipynb_to_py('./data_merge.ipynb', './data_merge.py')
