import numpy as np
from  autograd.engine import Variable,Executor,gradients,Node,log,exp,reduce_sum


x = Variable(name='x')
w = Variable(name='w')
y = Variable(name='y')

h = 1 / (1 + exp(-reduce_sum(w * x)))
L = y * log(h) + (1 - y) * log(1 - h)
w_grad, = gradients(L, [w])
##期望的结果
executor = Executor([L, w_grad])

y_val = 0
x_val = np.array([2, 3, 4])
w_val = np.random.random(3)

#动态度得到梯度
L_val, w_grad_val = executor.run(feed_dict={x: x_val, y: y_val, w: w_val})

#用numpy 计算 真是梯度
logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
expected_L_val = y_val * np.log(logistic) + (1 - y_val) * np.log(1 - logistic)
expected_w_grad = (y_val - logistic) * x_val

assert np.sum(np.abs(expected_w_grad - w_grad_val)) < 1E-9