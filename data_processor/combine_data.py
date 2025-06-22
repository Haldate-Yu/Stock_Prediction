import os
import pandas as pd


def merge_excel_files(root_dir, output_file):
    # 存储所有Excel文件数据的列表
    all_data = []

    # 遍历根目录下的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 查找当前目录下的所有Excel文件
        for filename in filenames:
            if filename.endswith('.xlsx'):
                file_path = os.path.join(dirpath, filename)
                try:
                    # 读取Excel文件
                    df = pd.read_excel(file_path)
                    all_data.append(df)
                    print(f"成功读取: {file_path}")
                except Exception as e:
                    print(f"读取失败 {file_path}: {e}")

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # 写入到输出文件
        combined_df.to_excel(output_file, index=False)
        print(f"已成功合并并写入到 {output_file}")
        return combined_df
    else:
        print("未找到任何Excel文件")
        return None


if __name__ == '__main__':
    # 使用示例
    root_directory = '../raw_data'  # 替换为你的目录路径
    output_excel = './combined_data.xlsx'  # 输出文件路径

    # 执行合并操作
    combined_data = merge_excel_files(root_directory, output_excel)

    # 查看合并后的数据基本信息
    if combined_data is not None:
        print(f"合并后的数据基本信息:")
        combined_data.info()
        print(f"数据包含 {len(combined_data)} 行和 {len(combined_data.columns)} 列")