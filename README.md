# autograd-engine
automatic differentiation engine  written in Numpy

You write the forward(), it does all backward() derivatives for you:
* Define-by-Run (just like PyTorch does) code:autograd-dynamic
* Define-and-Run (similar to the static graph of TensorFlow) code:autograd-static





自动微分(Automatic differentiation (AD)):利用计算图的形式对函数进行数字微分求导,是现在机器学习/深度学习的基石之一;


Automatic differentiation
只需要手动写forward(),无需实现backward():
* Define-by-Run: just like PyTorch does.  动态图,在运行时构建计算图,并进行数据前向与反向求导;
这种形式.相应的实现方式在autograd-dynamic;
Define-and-Run (similar to the static graph of TensorFlow)静态图,一开始就定义好计算图,每次运算只是feed 数据.相应的实现方式在autograd-static




## 自动微分介绍
传统求导方式:
1. 数值微分:
   $$f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}$$

利用数值逼近的方式求导;
2. 符号微分:
   $$\left(\frac{x^{2} \cos (x-7)}{\sin (x)}\right)^{\prime}=x \csc (x)(\cos (x-7)(2-x \cot (x))-x \sin (x-7))$$
具体得到某个函数的导数表达式来得到求导值


这两种方式在计算较高阶导数的的时候,复杂度和误差都会增加.


自动求导通过链式法则来进行计算图的构建与导数计算:
下面是一个例子:

$$\begin{array}{l}
w_{1}=x_{1}=2 \\
w_{2}=x_{2}=3 \\
w_{3}=w_{1} w_{2} \\
w_{4}=\sin \left(w_{1}\right) \\
w_{5}=w_{3}+w_{4}=y
\end{array}$$

其计算图构建的规则为每个运算为一个op算子操作,在图中为节点.最终的输出节点通过数据流的形式展示,如图所示:
![](http://image.haibin.online/2021-01-13,21:13:39.jpg)


前向计算:

![](http://image.haibin.online/2021-01-13,21:15:20.jpg)

对w1的反向求导:
![](http://image.haibin.online/2021-01-13,21:15:58.jpg)






    
