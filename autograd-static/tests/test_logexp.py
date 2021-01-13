import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node,log,exp,reduce_sum


class TestEngineLogExp(unittest.TestCase):
    def test_log_op(self):
        x1 = Variable(name="x1")
        y = log(x1)

        grad_x1, = gradients(y, [x1])

        executor = Executor([y, grad_x1])
        x1_val = 2 * np.ones(3)
        y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, np.log(x1_val))
        assert np.array_equal(grad_x1_val, 1 / x1_val)

    def test_log_two_vars(self):
        x1 = Variable(name="x1")
        x2 = Variable(name="x2")
        y = log(x1 * x2)

        grad_x1, grad_x2 = gradients(y, [x1, x2])

        executor = Executor([y, grad_x1, grad_x2])
        x1_val = 2 * np.ones(3)
        x2_val = 4 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, np.log(x1_val * x2_val))
        assert np.array_equal(grad_x1_val, x2_val / (x1_val * x2_val))
        assert np.array_equal(grad_x2_val, x1_val / (x1_val * x2_val))

    def test_exp_op(self):
        x1 = Variable(name="x1")
        y = exp(x1)

        grad_x1, = gradients(y, [x1])

        executor = Executor([y, grad_x1])
        x1_val = 2 * np.ones(3)
        y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, np.exp(x1_val))
        assert np.array_equal(grad_x1_val, np.exp(x1_val))

    def test_exp_mix_op(self):
        x1 = Variable(name="x1")
        x2 = Variable(name="x2")
        y = exp(log(x1 * x2) + 1)

        grad_x1, grad_x2 = gradients(y, [x1, x2])

        executor = Executor([y, grad_x1, grad_x2])
        x1_val = 2 * np.ones(3)
        x2_val = 4 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, np.exp(np.log(x1_val * x2_val) + 1))
        assert np.array_equal(grad_x1_val, y_val * x2_val / (x1_val * x2_val))
        assert np.array_equal(grad_x2_val, y_val * x1_val / (x1_val * x2_val))

    def test_reduce_sum(self):
        x1 = Variable(name="x1")
        y = reduce_sum(x1)

        grad_x1, = gradients(y, [x1])

        executor = Executor([y, grad_x1])
        x1_val = 2 * np.ones(3)
        y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, np.sum(x1_val))
        assert np.array_equal(grad_x1_val, np.ones_like(x1_val))

    def test_reduce_sum_mix(self):
        x1 = Variable(name="x1")
        y = exp(reduce_sum(x1))

        grad_x1, = gradients(y, [x1])

        executor = Executor([y, grad_x1])
        x1_val = 2 * np.ones(3)
        y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})
        expected_y_val = np.exp(np.sum(x1_val))
        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_y_val)
        assert np.array_equal(grad_x1_val, expected_y_val * np.ones_like(x1_val))

        y2 = log(reduce_sum(x1))
        grad_x2, = gradients(y2, [x1])
        executor2 = Executor([y2, grad_x2])
        y2_val, grad_x2_val = executor2.run(feed_dict={x1: x1_val})
        expected_y2_val = np.log(np.sum(x1_val))
        assert isinstance(y2, Node)
        assert np.array_equal(y2_val, expected_y2_val)
        assert np.array_equal(grad_x2_val, (1 / np.sum(x1_val)) * np.ones_like(x1_val))

    def test_mix_all(self):
        x1 = Variable(name="x1")
        y = 1 / (1 + exp(-reduce_sum(x1)))

        grad_x1, = gradients(y, [x1])

        executor = Executor([y, grad_x1])
        x1_val = 2 * np.ones(3)
        y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})
        expected_y_val = 1 / (1 + np.exp(-np.sum(x1_val)))
        expected_y_grad = expected_y_val * (1 - expected_y_val) * np.ones_like(x1_val)


        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_y_val)
        assert np.sum(np.abs(grad_x1_val - expected_y_grad)) < 1E-10

    def test_logistic(self):
        x1 = Variable(name="x1")
        w = Variable(name='w')
        y = 1 / (1 + exp(-reduce_sum(w * x1)))

        grad_w, = gradients(y, [w])

        executor = Executor([y, grad_w])
        x1_val = 3 * np.ones(3)
        w_val = 3 * np.zeros(3)
        y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
        expected_y_val = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
        expected_y_grad = expected_y_val * (1 - expected_y_val) * x1_val


        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_y_val)
        assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7

    def test_log_logistic(self):
        x1 = Variable(name="x1")
        w = Variable(name='w')
        y = log(1 / (1 + exp(-reduce_sum(w * x1))))

        grad_w, = gradients(y, [w])

        executor = Executor([y, grad_w])
        x1_val = 3 * np.ones(3)
        w_val = 3 * np.zeros(3)
        y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
        logistic = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
        expected_y_val = np.log(logistic)
        expected_y_grad = (1 - logistic) * x1_val


        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_y_val)
        assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7

    def test_logistic_loss(self):
        x = Variable(name='x')
        w = Variable(name='w')
        y = Variable(name='y')

        h = 1 / (1 + exp(-reduce_sum(w * x)))
        L = y * log(h) + (1 - y) * log(1 - h)
        w_grad, = gradients(L, [w])
        executor = Executor([L, w_grad])

        y_val = 0
        x_val = np.array([2, 3, 4])
        w_val = np.random.random(3)

        L_val, w_grad_val = executor.run(feed_dict={x: x_val, y: y_val, w: w_val})

        logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
        expected_L_val = y_val * np.log(logistic) + (1 - y_val) * np.log(1 - logistic)
        expected_w_grad = (y_val - logistic) * x_val



        assert expected_L_val == L_val
        assert np.sum(np.abs(expected_w_grad - w_grad_val)) < 1E-9