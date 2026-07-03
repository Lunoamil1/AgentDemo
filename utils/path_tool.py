"""为整个工程统一的绝对路径"""

import  os



def get_project_root()->str:
    """当前文件绝对路径"""
    current_file=os.path.abspath(__file__)

    current_dir = os.path.dirname(current_file)

    project_root = os.path.dirname(current_dir)

    return project_root




def get_abs_path(relative_path:str)->str:
    """获取相对路径的绝对路径"""
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/con.txt"))
