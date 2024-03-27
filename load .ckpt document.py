import pytorch_lightning as pl
 
# 定义一个 PyTorch Lightning 训练模块
class MyLightningModel(pl.LightningModule):
   def __init__(self):
       super().__init__()
       self.linear_layer = nn.Linear(10, 1)
       self.loss_function = nn.MSELoss()
 
   def forward(self, inputs):
       return self.linear_layer(inputs)
 
   def training_step(self, batch, batch_idx):
       features, targets = batch
       predictions = self(features)
       loss = self.loss_function(predictions, targets)
       self.log('train_loss', loss)
       return loss
 
# 初始化 PyTorch Lightning 模型
lightning_model = MyLightningModel()
 
# 配置 ModelCheckpoint 回调以定期保存最佳模型至 .ckpt 文件
checkpoint_callback = pl.callbacks.ModelCheckpoint(
   monitor='val_loss',
   filename='best-model-{epoch:02d}-{val_loss:.2f}',
   save_top_k=3,
   mode='min'
)
 
# 创建训练器并启动模型训练
trainer = pl.Trainer(
   callbacks=[checkpoint_callback],
   max_epochs=10
)
trainer.fit(lightning_model)
 
# 从 .ckpt 文件加载最优模型权重
best_model = MyLightningModel.load_from_checkpoint(checkpoint_path='best-model.ckpt')
 
# 使用加载的 .ckpt 文件中的模型进行预测
sample_input = torch.randn(1, 10)
predicted_output = best_model(sample_input)
print(predicted_output)