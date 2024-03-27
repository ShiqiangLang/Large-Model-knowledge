# 假设 model 是一个 PyTorch Lightning 模型实例
model = MyLightningModel()
 
# 保存模型权重
torch.save(model.state_dict(), 'lightning_model.pth')
 
# 加载到一个新的 PyTorch 模型实例
new_model = MyLightningModel()
new_model.load_state_dict(torch.load('lightning_model.pth'))
 
# 或者加载到一个普通的 PyTorch Module 实例（假设结构一致）
plain_pytorch_model = MyPlainPytorchModel()
plain_pytorch_model.load_state_dict(torch.load('lightning_model.pth'))