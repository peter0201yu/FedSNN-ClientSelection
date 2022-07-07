import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class Simple_CF10_BNTT(nn.Module):
    def __init__(self, timesteps=10, leak_mem=0.95, img_size=32,  num_cls=10):
        super(Simple_CF10_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        print (">>>>>>>>>>>>>>>>>>> Simple Model >>>>>>>>>>>>>>>>>>>>>>")
        print ("{} time step per batchnorm for a client".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=8)

        self.fc1 = nn.Linear(1024, 64, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(64, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt_fc]
        self.pool_list = [False, self.pool1]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv_list = [mem_conv1, mem_conv2]

        mem_fc1 = torch.zeros(batch_size, 64).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.timesteps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage

class VGG5_CF10_NoBNTT(nn.Module):
    def __init__(self, timesteps, leak_mem=0.95, img_size=32, num_cls=10, input_dim=3):
        super(VGG5_CF10_NoBNTT, self).__init__()
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = timesteps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.arch = "SNN"
        print(">>>>>>>>>>>>>>>>>>> VGG5_Direct Coding >>>>>>>>>>>>>>>>>>>>>>")
        # cifar10 & cifar 100
        if input_dim == 3:
            dim = 8
        # mnist
        elif input_dim == 1:
            dim = 5

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * dim * dim, 1024, bias=False)
        self.fc2 = nn.Linear(1024, num_cls, bias=False)

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=3)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=3)

        self.conv_list = [self.conv1, self.conv2, self.conv3]

        self.pool_list = [self.pool1, self.pool2]

        self.fc_list = [self.fc1, self.fc2]

    def forward(self, inp):

        batch_size = inp.size(0)

        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, (self.img_size) // 2, (self.img_size) // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, (self.img_size) // 2, (self.img_size) // 2).cuda()

        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        mem_fc_list = [mem_fc1, mem_fc2]

        # Direct coding - static input from conv1

        static_input = self.conv1(inp)

        for t in range(self.num_steps):
            # Charging and firing (lif for conv1)
            mem_conv_list[0] = self.leak_mem * mem_conv_list[0] + (1 - self.leak_mem) * static_input
            mem_thr = (mem_conv_list[0] / self.conv_list[0].threshold) - 1.0
            out = self.spike_fn(mem_thr)

            # Soft reset
            rst = torch.zeros_like(mem_conv_list[0]).cuda()
            rst[mem_thr > 0] = self.conv_list[0].threshold
            mem_conv_list[0] = mem_conv_list[0] - rst
            out_prev = out.clone()

            # Pooling
            out = self.pool_list[0](out_prev)
            out_prev = out.clone()

            mem_conv_list[1] = self.leak_mem * mem_conv_list[1] + (1 - self.leak_mem) * self.conv2(out_prev)
            mem_thr = (mem_conv_list[1] / self.conv_list[1].threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv_list[1]).cuda()
            rst[mem_thr > 0] = self.conv_list[1].threshold
            mem_conv_list[1] = mem_conv_list[1] - rst
            out_prev = out.clone()

            # print ("aa", out_prev.sum())

            mem_conv_list[2] = self.leak_mem * mem_conv_list[2] + (1 - self.leak_mem) * self.conv3(out_prev)
            mem_thr = (mem_conv_list[2] / self.conv_list[2].threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv_list[2]).cuda()
            rst[mem_thr > 0] = self.conv_list[2].threshold
            mem_conv_list[2] = mem_conv_list[2] - rst
            out_prev = out.clone()

            # print ("bb",out_prev.sum())

            # Pooling
            out = self.pool_list[1](out_prev)
            out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            for i in range(len(self.fc_list) - 1):
                mem_fc_list[i] = self.leak_mem * mem_fc_list[i] + (1 - self.leak_mem) * self.fc_list[i](out_prev)
                mem_thr = (mem_fc_list[i] / self.fc_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)

                rst = torch.zeros_like(mem_fc_list[i]).cuda()
                rst[mem_thr > 0] = self.fc_list[i].threshold
                mem_fc_list[i] = mem_fc_list[i] - rst
                out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage
