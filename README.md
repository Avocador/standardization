# standardization
**Learn to make the project more standardized, and norms are important.**

Main reference materials：[github]:https://github.com/achernodub/targer

![](https://res.cloudinary.com/avocador/image/upload/v1637676659/project/project_tgvjno.png)




---

some tips about `targer`:
- 通过`import`的形式从指定文件夹中加载py文件内容，注意，其他子文件的索引路径应与main.py的位置保持一致。
- 通过先在`__init__()`函数创建self.variable_name，再将self.variable_name进行赋值的方式，将变量以属性的形式完成集成。
- 注意传参的形式，尤其是args参数如何引入子索引文件。




---

thoughts:
- layer或model中的传参，可以在`__init__()`时通过固定为self属性的方式传递给`forward()`，这样`forward()`无需重复传参。`model = ModelFactory.create(args, ...)`，重复传参形式为`outputs = model(input_batch, args)`，非重复传参形式为`outputs = model(input_batch)`。
