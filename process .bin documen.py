 # 假设已经从.bin文件中读取到了模型权重数据
 weights_data = load_binary_weights("weights.bin")
 
 # 手动初始化模型并加载权重
 model = TheModelClass(*args, **kwargs)
 for name, param in model.named_parameters():
     if name in weights_mapping:  # 需要预先知道权重映射关系
         param.data.copy_(weights_data[weights_mapping[name]])