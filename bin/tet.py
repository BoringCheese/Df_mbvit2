from local_configs import C

def main():
    print("训练周期数：", C.nepochs)
    print("数据集名称：", C.dataset_name)

if __name__ == "__main__":
    main()
