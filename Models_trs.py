import torch
import torch.nn as nn
import torch.nn.functional as F

K = 20

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = torch.device("cuda:" + str(x.get_device()) if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """
    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = False if args.model == 'dgcnn' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = torch.device("cuda:" + str(x.get_device()) if self.args.cuda else "cpu")

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == "dgcnn":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        if self.args.model == "dgcnn_trs":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class PointNet(nn.Module):
    def __init__(self, args, num_class=10):
        super(PointNet, self).__init__()
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3)

        logits = {}

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x4)
        x5, _ = torch.max(x, dim=2, keepdim=False)
        x = x5.squeeze(dim=2)  # batchsize*1024

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat.squeeze(dim=3), x5.repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits

class DGCNN_trs3(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCNN_trs3, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(256)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64*2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64*2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128*2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        self.C = classifier2(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        # x0 = get_graph_feature(x, self.args, k=self.k)
        # transformd_x0 = self.input_transform_net(x0)
        # x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x_sa1 = self.sa1(x1)

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x_sa2 = self.sa2(x2)

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x_sa3 = self.sa3(x3)

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x_sa4 = self.sa4(x4)

        x_cat = torch.cat((x_sa1, x_sa2, x_sa3, x_sa4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x5_1, x5_2), 1)

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5_1.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits

class DGCNN_trs3_1(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCNN_trs3_1, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(256)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64*2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64*2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128*2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        self.C = classifier2(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x_sa1 = self.sa1(x1)

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x_sa2 = self.sa2(x2)

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x_sa3 = self.sa3(x3)

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x_sa4 = self.sa4(x4)

        x_cat = torch.cat((x_sa1, x_sa2, x_sa3, x_sa4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x5_1, x5_2), 1)

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5_1.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits


class DGCNN_trs4(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCNN_trs4, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(256)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64*2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64*2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128*2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        self.C = classifier2(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        # x0 = get_graph_feature(x, self.args, k=self.k)
        # transformd_x0 = self.input_transform_net(x0)
        # x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.sa1(x1)

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.sa2(x2)

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = self.sa3(x3)

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = self.sa4(x4)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x5_1, x5_2), 1)

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5_1.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits

class classifier(nn.Module):
    def __init__(self, args, num_class=10):
        super(classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(1024* 2, 512, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits

class classifier2(nn.Module):
    def __init__(self, args, num_class=10):
        super(classifier2, self).__init__()

        self.mlp1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bnm1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = nn.Linear(512, 256)
        self.bnm2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = F.leaky_relu(self.bnm1(self.mlp1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x2 = F.leaky_relu(self.bnm2(self.mlp2(x)), negative_slope=0.2)
        x2 = self.dp2(x2)
        logits = self.mlp3(x2)
        return logits

class RegionReconstruction(nn.Module):
    """
    Region Reconstruction Network - Reconstruction of a deformed region.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """
    def __init__(self, args, input_size):
        super(RegionReconstruction, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x.permute(0, 2, 1)