import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.nn import functional as F


class HamRevBlock(nn.Module):
    '''
    The basic building block of the Hamiltonian network. Implements the two-layer Hamiltonian architecture
    proposed in 'Reversible Architectures for Arbitrarily Deep Residual Neural Networks'. The F,G blocks are
    residual functions that can be used to reverse the architecture in the backward pass.
    '''
    def __init__(self, f_block, g_block, split_along_dim=1):
        super(HamRevBlock, self).__init__()
        self.f_block = f_block
        self.g_block = g_block
        self.split_along_dim = split_along_dim

    def forward(self, x):
        '''

        :param x: Input tensor.
        :return: Output tensor of the same shape as the input tensor.
        '''
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        with torch.no_grad():
            y1 = x1 + self.f_block(x2)
            y2 = x2 + self.g_block(y1)
        return torch.cat([y1, y2], dim=self.split_along_dim)

    def backward(self, y, dy):
        '''

        :param y: Output of the Hamiltonian reversible block
        :param dy: Derivatives of the output.
        :return: Tuple of the block's input and input derivative.
        '''

        # Split into two channels
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        del y
        assert (not y1.requires_grad), "y1 must already be detached"
        assert (not y2.requires_grad), "y2 must already be detached"
        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_along_dim)
        del dy
        assert (not dy1.requires_grad), "dy1 must not require grad"
        assert (not dy2.requires_grad), "dy2 must not require grad"
        y1.requires_grad = True
        y2.requires_grad = True

        # Back propagation for G function
        with torch.enable_grad():
            g_y1 = self.g_block(y1)
            g_y1.backward(dy2)

        # Compute input x2 and input derivative dx1 out of G function and output y2
        # and output derivative dy1
        with torch.no_grad():
            x2 = y2 - g_y1
            del y2, g_y1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        # Backpropagation for F function
        with torch.enable_grad():
            x2.requires_grad = True
            f_x2 = self.f_block(x2)
            f_x2.backward(dx1)

        # Compute input x1 and input derivative dx2 out of F function and output y1
        # and output derivative dy2
        with torch.no_grad():
            x1 = y1 - f_x2
            del y1, f_x2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Concatenate the two channels
            x = torch.cat([x1, x2.detach()], dim=self.split_along_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_along_dim)

        return x, dx


class RevUnitFunction(Function):

    @staticmethod
    def forward(ctx, x, reversible_blocks):
        '''

        Preforms forward pass of a Hamiltonain reversible unit with autograd framework.

        :param ctx: Autograd context.
        :param x: Input tensor.
        :param reversible_blocks: List of Hamiltonian reversible blocks.
        :return: Output tensor.
        '''
        assert (isinstance(reversible_blocks, nn.ModuleList))
        for block in reversible_blocks:
            assert (isinstance(block, HamRevBlock))
            x = block(x)
        ctx.y = x.detach()
        ctx.reversible_blocks = reversible_blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        '''

         Preforms backward pass of a Hamiltonain reversible unit with autograd framework.

        :param ctx: Autograd context.
        :param dy: Derivatives of the output.
        :return: Derivatives of the input.
        '''
        y = ctx.y
        #del ctx.y
        for i in range(len(ctx.reversible_blocks)-1, -1, -1):
            y, dy = ctx.reversible_blocks[i].backward(y, dy)
        #del ctx.reversible_blocks
        return dy, None, None


class RevBlockUnit(nn.Module):
    '''

    The Hamiltonian reversible unit is a sequence of arbitrarily many Hamiltonian reversible blocks. The unit is
    reversible, therefore the activations are saved only at the end of the unit. While preforming backpropagation,
    the optimizer leverages the reversibility of the unit to save memory.
    '''
    def __init__(self, reversible_blocks):
        super(RevBlockUnit, self).__init__()
        assert (isinstance(reversible_blocks,nn.ModuleList))
        for block in reversible_blocks:
            assert (isinstance(block, HamRevBlock))
        self.reversible_blocks = reversible_blocks

    def forward(self, x):
        '''

        :param x: Input tensor.
        :return: Output tensor.
        '''
        x = RevUnitFunction.apply(x, self.reversible_blocks)
        return x


class InceptionRevBlock(nn.Module):
    def __init__(self, n_channels, sign):
        super(InceptionRevBlock, self).__init__()
        assert (sign == -1 or sign == 1)
        self.sign = sign
        n_channels_1 = int(n_channels // 4)
        n_channels_2 = int(n_channels // 4)
        n_channels_3 = int(n_channels // 4)
        self.conv11 = nn.Conv2d(n_channels_1, n_channels_1, 1, padding=1, bias=True)  # Conv 1x1 for Conv 1x1 branch
        self.conv13 = nn.Conv2d(n_channels_2, n_channels_2, 1, padding=1, bias=True)  # Conv 1x1 for Conv 3x3 branch
        self.conv15 = nn.Conv2d(n_channels_3, n_channels_3, 1, padding=1, bias=True)  # Conv 1x1 for Conv 5x5 branch
        self.conv3 = nn.Conv2d(n_channels_2, n_channels_2, 3, padding=1, bias=True)  # Conv 3x3 for Conv 3x3 branch
        self.conv5 = nn.Conv2d(n_channels_3, n_channels_3, 5, padding=1, bias=True)  # Conv 5x5 for Conv 5x5 branch

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        branch1 = F.conv2d(x1, self.conv11.weight, bias=self.conv11.bias)
        branch2_1 = F.conv2d(x2, self.conv13.weight, bias=self.conv13.bias)
        branch2_2 = F.conv2d(branch2_1, self.conv3.weight, bias=self.conv3.bias, padding=1)
        branch3_1 = F.conv2d(x3, self.conv15.weight, bias=self.conv15.bias)
        branch3_2 = F.conv2d(branch3_1, self.conv5.weight, bias=self.conv5.bias, padding=2)

        x = torch.cat([branch1, branch2_2, branch3_2, x4], 1)
        x = F.relu(x)

        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        branch1_T = F.conv_transpose2d(x1, self.conv11.weight, bias=self.conv11.bias)
        branch2_1_T = F.conv_transpose2d(x2, self.conv13.weight, bias=self.conv13.bias)
        branch2_2_T = F.conv_transpose2d(branch2_1_T, self.conv3.weight, bias=self.conv3.bias, padding=1)
        branch3_1_T = F.conv_transpose2d(x3, self.conv15.weight, bias=self.conv15.bias)
        branch3_2_T = F.conv_transpose2d(branch3_1_T, self.conv5.weight, bias=self.conv5.bias, padding=2)
        x = torch.cat([branch1_T, branch2_2_T, branch3_2_T, x4], 1)
        x = self.sign * x
        return x


class ResidualFunctionBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, sign):
        super(ResidualFunctionBlock, self).__init__()
        assert (sign == -1 or sign == 1)
        self.sign = sign
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size, padding=1, bias=True)

    def forward(self, x):
        x = F.conv2d(x, self.conv.weight, bias=self.conv.bias)
        x = F.relu(x)
        x = F.conv_transpose2d(x, self.conv.weight, bias=self.conv.bias)
        x = self.sign * x
        return x


class HamiltonianOriginalNetwork(nn.Module):
    def __init__(self):
        super(HamiltonianOriginalNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_conv = nn.Conv2d(3, 32, 3)
        self.dropout1 = nn.Dropout2d(0.6)
        self.dropout2 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        blocks1 = []
        blocks2 = []
        blocks3 = []
        block_num_per_unit = 18
        for i in range(block_num_per_unit):
            f_func1 = ResidualFunctionBlock(16, 3, sign=1)
            g_func1 = ResidualFunctionBlock(16, 3, sign=-1)
            f_func2 = ResidualFunctionBlock(32, 3, sign=1)
            g_func2 = ResidualFunctionBlock(32, 3, sign=-1)
            f_func3 = ResidualFunctionBlock(64, 3, sign=1)
            g_func3 = ResidualFunctionBlock(64, 3, sign=-1)
            blocks1.append(HamRevBlock(f_func1, g_func1))
            blocks2.append(HamRevBlock(f_func2, g_func2))
            blocks3.append(HamRevBlock(f_func3, g_func3))

        self.hamiltonian_unit1 = RevBlockUnit(nn.ModuleList(blocks1))
        self.hamiltonian_unit2 = RevBlockUnit(nn.ModuleList(blocks2))
        self.hamiltonian_unit3 = RevBlockUnit(nn.ModuleList(blocks3))
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(128 * 2 * 2, 10)

    def forward(self, x):
        #x = self.dropout1(x)
        x = self.init_conv(x)
        x = self.bn1(x)
        x = self.hamiltonian_unit1(x)
        #x = self.bn1(x)
        x = self.pool1(x)
        x = self.zero_pad(x, x.shape)
        x = self.hamiltonian_unit2(x)
        #x = self.bn2(x)
        x = self.pool2(x)
        x = self.zero_pad(x, torch.Size([x.shape[0], 128-x.shape[1], x.shape[2], x.shape[3]]))
        x = self.hamiltonian_unit3(x)
        #x = self.bn3(x)
        x = self.pool3(x)

        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.dropout2(x)
        x = self.fc(x)
        return x

    def zero_pad(self, x, padding_shape):
        zero_padding = torch.zeros(padding_shape).to(self.device)
        x = torch.cat([x, zero_padding], dim=1)
        return x


class HamiltonianInceptionNetwork(nn.Module):
    def __init__(self):
        super(HamiltonianInceptionNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_conv = nn.Conv2d(3, 32, 3)
        self.dropout1 = nn.Dropout2d(0.6)
        self.dropout2 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)

        blocks1 = []
        blocks2 = []
        blocks3 = []
        block_num_per_unit = 18
        for i in range(block_num_per_unit):
            f_func1 = InceptionRevBlock(16, sign=1)
            g_func1 = InceptionRevBlock(16, sign=-1)
            f_func2 = InceptionRevBlock(32, sign=1)
            g_func2 = InceptionRevBlock(32, sign=-1)
            f_func3 = InceptionRevBlock(64, sign=1)
            g_func3 = InceptionRevBlock(64, sign=-1)
            blocks1.append(HamRevBlock(f_func1, g_func1))
            blocks2.append(HamRevBlock(f_func2, g_func2))
            blocks3.append(HamRevBlock(f_func3, g_func3))

        self.hamiltonian_unit1 = RevBlockUnit(nn.ModuleList(blocks1))
        self.hamiltonian_unit2 = RevBlockUnit(nn.ModuleList(blocks2))
        self.hamiltonian_unit3 = RevBlockUnit(nn.ModuleList(blocks3))
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(128 * 2 * 2, 10)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.bn1(x)
        x = self.hamiltonian_unit1(x)
        x = self.pool1(x)
        x = self.zero_pad(x, x.shape)
        x = self.hamiltonian_unit2(x)
        x = self.pool2(x)
        x = self.zero_pad(x, torch.Size([x.shape[0], 128-x.shape[1], x.shape[2], x.shape[3]]))
        x = self.hamiltonian_unit3(x)
        x = self.pool3(x)

        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.dropout2(x)
        x = self.fc(x)
        return x

    def zero_pad(self, x, padding_shape):
        zero_padding = torch.zeros(padding_shape).to(self.device)
        x = torch.cat([x, zero_padding], dim=1)
        return x
