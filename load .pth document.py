 # 保存模型状态字典
 torch.save(model.state_dict(), "model.pth")
 
 # 加载模型状态字典到已有模型结构中
 model = TheModelClass(*args, **kwargs)
 model.load_state_dict(torch.load("model.pth"))
 
 # 或者保存整个模型，包括结构
 torch.save(model, "model.pth")
 
 # 加载整个模型
 model = torch.load("model.pth", map_location=device)