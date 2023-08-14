import torch

class Upsample1d(torch.nn.Module):
    def __init__(self, dim, dim_out, scale_factor, mode="nearest", conv=True):
        super(Upsample1d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

        self.upsample = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=scale_factor, mode=mode)])
        if conv:
            self.upsample.append(torch.nn.Conv1d(dim, dim_out, 3, padding=1))

    def forward(self, x):
        for layer in self.upsample:
            x = layer(x)
        return x
    
class Block1d(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = torch.nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResBlock1d(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        self.block1 = Block1d(dim, dim_out, groups)
        self.block2 = Block1d(dim_out, dim_out, groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet1d(torch.nn.Module):
    def __init__(self, in_ch, dim, out_ch, num_time_steps, groups=8, time_embedding="naive"):
        super(UNet1d, self).__init__()

        self.time_embedding = time_embedding
        if time_embedding == "naive":
            self.init_conv = torch.nn.Conv1d(in_ch + 1, dim, 7, padding = 3)
        else:
            self.init_conv = torch.nn.Conv1d(in_ch, dim, 7, padding = 3)

        self.num_time_steps = num_time_steps

        dims = [dim, 2*dim, 4*dim, 8*dim]
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        for i in range(len(dims) - 1):
            self.downs.append(torch.nn.ModuleList([
                ResBlock1d(dims[i], dims[i+1], groups),
                ResBlock1d(dims[i+1], dims[i+1], groups),
                torch.nn.AvgPool1d(2),
            ]))

        self.mid_block1 = ResBlock1d(dims[-1], dims[-1], groups)
        self.mid_block2 = ResBlock1d(dims[-1], dims[-1], groups)    

        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(torch.nn.ModuleList([
                Upsample1d(dims[i-1], dims[i-1], 2, conv=False),
                ResBlock1d(2 * dims[i], dims[i-1], groups),
                ResBlock1d(2 * dims[i-1], dims[i-1], groups),
            ]))

        self.final_res_block = ResBlock1d(2 * dim, dim, groups)
        self.final_conv = torch.nn.Conv1d(dim, out_ch, 1)

    def forward(self, x, t):
        if self.time_embedding == "naive":
            o = t.view(-1, 1, 1).expand(-1, 1, x.shape[-1]).to(x.device) / self.num_time_steps
            o = torch.cat([o, x], dim=1)
        else:
            o = x

        o = self.init_conv(o)
        r = o.clone()

        h = []
        for block1, block2, pool in self.downs:
            h.append(o)
            o = block1(o)

            h.append(o)
            o = block2(o)

            o = pool(o)

        o = self.mid_block1(o)
        o = self.mid_block2(o)

        for upsample, block1, block2 in self.ups:
            o = upsample(o)

            o = torch.cat([h.pop(), o], dim=1)
            o = block1(o)

            o = torch.cat([h.pop(), o], dim=1)
            o = block2(o)
        
        o = torch.cat([r, o], dim=1)
        o = self.final_res_block(o)
        o = self.final_conv(o)

        return o
