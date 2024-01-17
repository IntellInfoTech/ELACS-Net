import torch
import numpy as np
import torch.nn as nn

import config


class DNNs(nn.Module):
    def __init__(self, num_layers):     
        super(DNNs, self).__init__()
        self.num_layers = num_layers
        self.layer_in = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ELU()
                )
            )
        self.layer_out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.layer_in(x)
        for i in range(self.num_layers):
            y = self.layers[i](y)
        y = self.layer_out(y)
        return x - y


class ELACS_Net(nn.Module):
    def __init__(self):
        super(ELACS_Net, self).__init__()
        self.block_size = config.para.block_size
        self.NUM_ITERATIONS = config.para.step_num
        self.idx = self.block_size // 32
        self.act = nn.ReLU()

        self.DNNs_num_layers = config.para.num_layers   

        self.n = 32 ** 2
        self.m = int(config.para.rate * self.n)
        init_phi = np.random.normal(0.0, (1 / self.n) ** 0.5, size=(self.m, self.n))
        
        self.phi = nn.Parameter(torch.from_numpy(init_phi).float(), requires_grad=True)
        self.omega = nn.Parameter(torch.from_numpy(np.transpose(init_phi)).float(), requires_grad=True)
        
        self.alpha = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.var_theta = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        self.a = 0.9
        self.one_step = 0.5
        self.steps = []
        for i in range(self.NUM_ITERATIONS):
            self.register_parameter("step_" + str(i),
                                    nn.Parameter(torch.tensor(self.one_step), requires_grad=True))
            self.steps.append(eval("self.step_" + str(i)))
        self.DNNs_transform_f = nn.ModuleList()
        for i in range(self.NUM_ITERATIONS):
            self.DNNs_transform_f.append(DNNs(self.DNNs_num_layers))    
        self.DNNs_transform_h = nn.ModuleList()
        for i in range(self.NUM_ITERATIONS):
            self.DNNs_transform_h.append(DNNs(self.DNNs_num_layers))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        y = self.sampling(inputs)
        recon = self.recon(y, batch_size)
        return recon

    def sampling(self, inputs):
        inputs = inputs.to(config.para.device)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=32, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=32, dim=2), dim=0)
        inputs = inputs.reshape(-1, 32 ** 2).transpose(0, 1)
        y = torch.matmul(self.phi, inputs)
        return y

    def recon(self, y, batch_size):
        x = torch.sqrt(1 + self.alpha) * torch.matmul(self.omega, y)
        x = x.transpose(0, 1).reshape(-1, 1, 32, 32)
        x = self.DNNs_transform_f[0](x)
        x = x.reshape(-1, 32 ** 2).transpose(0, 1)

        v = 0
        theta_plus = 1 / torch.sqrt(1 + self.alpha) * torch.cat(
            (self.phi, torch.eye(self.n, self.n).to(config.para.device)), 0)
        y = torch.cat((y, torch.zeros(self.n, y.size(1)).to(config.para.device)), 0)
        for k in range(1, self.NUM_ITERATIONS):
            v_prev = v
            v = self.a * v_prev - self.steps[k] * torch.matmul(
                theta_plus.transpose(0, 1), (torch.matmul(theta_plus, x) - y))
            g = x + self.a * v - self.steps[k] * torch.matmul(
                theta_plus.transpose(0, 1), (torch.matmul(theta_plus, x) - y))
            g = g.transpose(0, 1).reshape(-1, 1, 32, 32)
            g = torch.cat(torch.split(g, split_size_or_sections=batch_size * self.idx, dim=0), dim=2)
            g = torch.cat(torch.split(g, split_size_or_sections=batch_size, dim=0), dim=3)

            tran = self.DNNs_transform_f[k](g)
            tran = torch.sign(tran) * self.act(
                torch.abs(tran) - self.var_theta * self.steps[k] *
                (1 - self.alpha) / (torch.sqrt(1 + self.alpha)))
            x = self.DNNs_transform_h[k](tran)

            x = torch.cat(torch.split(x, split_size_or_sections=32, dim=3), dim=0)
            x = torch.cat(torch.split(x, split_size_or_sections=32, dim=2), dim=0)
            x = x.reshape(-1, 32 ** 2).transpose(0, 1)

        x_hat = x.transpose(0, 1).reshape(-1, 1, 32, 32)
        x_hat = self.DNNs_transform_h[0](x_hat) / torch.sqrt(1 + self.alpha)
        x_hat = torch.cat(torch.split(x_hat, split_size_or_sections=batch_size * self.idx, dim=0), dim=2)
        x_hat = torch.cat(torch.split(x_hat, split_size_or_sections=batch_size, dim=0), dim=3)
        return x_hat, None
