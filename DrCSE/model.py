import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class Dropout(nn.Module):
    def __init__(self, dropout_rate, upsampling_num):
        super(Dropout, self).__init__()

        self.dropout_rate = dropout_rate
        self.upsampling_num = upsampling_num

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, source_embedding):

        new_samples = []
        for i in range(self.upsampling_num):
            new_source_embedding = self.dropout(source_embedding)
            new_samples.append(new_source_embedding)
        output = torch.Tensor(output)
        return output


class ClassPrediction(nn.Module):

    def __init__(self, opt):
        super(ClassPrediction, self).__init__()

        self.class_prediction = nn.Sequential(
            nn.Linear(opt.n_node, opt.n_hidden),
            nn.Tanh(),
            nn.Linear(opt.n_hidden, opt.n_classes),
            nn.Softmax(dim=1)
        )

        self.criterion = nn.CrossEntropyLoss()
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, graph_representation, target):
        output = self.class_prediction(graph_representation)
        loss = self.criterion(output, target)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1.0 - label) * torch.pow(euclidean_distance, 2) + (
            label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class GGNN(nn.Module):
    def __init__(self, opt):
        super(GGNN, self).__init__()

        # assert (opt.state_dim >= opt.annotation_dim, 'state_dim must be no less than annotation_dim')
        self.is_training_ggnn = opt.is_training_ggnn
        self.state_dim = opt.state_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.n_classes = opt.n_classes
        self.dropout_rate = opt.dropout_rate
        self.upsampling_num = opt.upsamping_num

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(
            self.state_dim, self.n_node, self.n_edge_types)

        self.Dropout = Dropout(self.dropout_rate, self.upsampling_num)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LeakyReLU(),
            nn.Linear(self.state_dim, 1),
            nn.Tanh(),
        )

        self.soft_attention = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LeakyReLU(),
            nn.Linear(self.state_dim, 1),
            nn.Sigmoid(),

        )

        self.class_prediction = nn.Sequential(
            nn.Linear(opt.state_dim, opt.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(opt.n_hidden, opt.n_classes),
            nn.Softmax(dim=1)
        )

        # self.class_prediction = nn.Sequential(
        #     nn.Linear(opt.n_node, opt.n_classes),
        #     nn.Softmax(dim=1)
        # )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    # def side_forward(self, prop_state, annotation, A):
    #     for i_step in range(self.n_steps):
    #         in_states = []
    #         out_states = []
    #         for i in range(self.n_edge_types):
    #             in_states.append(self.in_fcs[i](prop_state))
    #             out_states.append(self.out_fcs[i](prop_state))
    #         in_states = torch.stack(in_states).transpose(0, 1).contiguous()
    #         in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
    #         out_states = torch.stack(out_states).transpose(0, 1).contiguous()
    #         out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

    #         prop_state = self.propogator(in_states, out_states, prop_state, A)

    #     output = self.out(prop_state)

    #     soft_attention_ouput = self.soft_attention(prop_state)
    #     output = torch.mul(output,soft_attention_ouput)
    #     output = output.sum(2)
    #     return output

    # def forward(self, left_prop_state, left_annotation, left_A, right_prop_state, right_annotation, right_A):
    #     left_output = self.side_forward(left_prop_state, left_annotation, left_A)
    #     right_output = self.side_forward(right_prop_state, right_annotation, right_A)

    #     left_output = self.feature_representation(left_output)
    #     right_output = self.feature_representation(right_output)

    #     if self.opt.loss == 1:
    #         return left_output, right_output
    #     else:
    #         concat_layer = torch.cat((left_output, right_output),1)
    #         output = self.fc_output(concat_layer)
    #         print(output)
    #         return output

    def forward(self, prop_state, A):
        # print(prop_state.shape)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))

            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node *
                                       self.n_edge_types, self.state_dim)

            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node *
                                         self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        # print("Prop state : " + str(prop_state.shape))
        # output = self.out(prop_state)
        # print("Out : " + str(output.shape))

        soft_attention_ouput = self.soft_attention(prop_state)
        # print("Soft : " + str(soft_attention_ouput.shape))
        output = torch.mul(prop_state, soft_attention_ouput)
        # print("Out : " + str(output.shape))
        output = output.sum(1)
        # print(output.shape)
        if self.is_training_ggnn == True:
            output = self.class_prediction(output)
        # print(output.shape)

        return output
