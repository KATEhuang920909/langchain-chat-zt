## 指定制定列的csv文件加载器


import zipfile
import tarfile
import py7zr
import os


class PKGFileLoader:
    def __init__(self, file_path: str,extract_path: str ):
        self.file_path = file_path
        self.extract_path = extract_path

    import os

    def load(self, password=None) :
        """
        自动识别压缩包类型并解压
        :param file_path: 压缩文件路径
        :param extract_to: 解压目录（默认同文件名目录）
        :param password: 密码（用于加密压缩包）
        """
        # if extract_to is None:
        #     extract_to = os.path.splitext(self.file_path)[0]
        # os.makedirs(extract_to, exist_ok=True)

        ext = os.path.splitext(self.file_path)[-1].lower()
        info = ""
        try:
            if self.file_path.endswith('.zip'):
                with zipfile.ZipFile(self.file_path, 'r') as z:
                    if password:
                        z.setpassword(password.encode())
                    z.extractall(self.extract_path)
                info += f"✅ ZIP 解压成功！"

            elif self.file_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(self.file_path, 'r:gz') as t:
                    t.extractall(self.extract_path)
                info += f"✅ TAR.GZ 解压成功！"

            elif self.file_path.endswith('.7z'):
                with py7zr.SevenZipFile(self.file_path, mode='r', password=password) as z:
                    z.extractall(self.extract_path)
                info += f"✅ 7Z 解压成功！"

            else:
                info += f"❌ 不支持的格式: {ext}"
                os.rmdir(self.extract_path)
                return info, []

            info += f"📁 文件已解压到: {self.extract_path}"
            folders, files_path, files = self.get_relative_paths(self.extract_path)

            return info, [folders, files_path, files]

        except Exception as e:
            info = f"❌ 解压失败 {self.file_path}: {str(e)}"
            return info, []

    def get_relative_paths(self, directory):
        # 存储文件和文件夹的相对路径
        files = []
        folders = []
        files_path = []

        # 遍历目录
        for root, dirs, filenames in os.walk(directory):
            # # 遍历文件夹
            for dir_name in dirs:
                # 获取文件夹的相对路径
                relative_path = os.path.relpath(os.path.join(root, dir_name), directory)
                folders.append(relative_path)

            # 遍历文件
            for filename in filenames:
                # 获取文件的相对路径
                relative_path = os.path.relpath(os.path.join(root, filename), directory)
                files.append(relative_path.split("\\")[-1])
                files_path.append(relative_path)
        return folders, files_path, files


if __name__ == '__main__':
    # 示例调用
    path = "D:/work/中台/中台安全运营工具/test/中台检查文档需求清单@充电桩场景精细化选址能力v1.zip"
    # path = "D:/work/中台/中台安全运营工具/test/中台检查文档需求清单@充电桩场景精细化选址能力v1"
    loader = PKGFileLoader(path,"D:/work/中台/中台安全运营工具/test/中台检查文档需求清单@充电桩场景精细化选址能力v1")
    folders, files_path, files = loader.get_relative_paths("D:/work/中台/中台安全运营工具/test/中台检查文档需求清单@充电桩场景精细化选址能力v1")
    print("folders",folders)
    print("files_path", files_path)
    print("folders", files)