import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
from tqdm import tqdm
# from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import gc
import math
import pickle

def calc_Jac(*params, model):
    cells = len(params) > 2
    if cells:
        inputs = params[0]
        states = (params[1], params[2])
        h0, c0 = states
    else:
        inputs, states = params  # inputs should be a single time step with batch_size entries
        h0 = states
    num_layers, batch_size, hidden_size = h0.shape
    _, seq_len, input_size = inputs.shape
    L = num_layers * hidden_size

    # Feed forward into network
    if cells:
        model_out, states_out = model(inputs, states)
        hn, cn = states_out
    else:
        model_out, hn = model(inputs, states)

    # Flatten output from different layers of RNN (needed for J calculation)
    hn_flat = torch.reshape(torch.transpose(hn, 0, 1), (batch_size, 1, L))

    J = torch.zeros(batch_size, L, L)  # placeholder
    for i in range(L):
        hn_flat[:, :, i].backward(torch.ones_like(hn_flat[:, :, i]), retain_graph=True)
        der = h0.grad
        der = torch.reshape(torch.transpose(der, 0, 1), (batch_size, L))
        J[:, i, :] = der
        h0.grad.zero_()
    return J


def oneStep(*params, model, rec_layer='lstm'):
    # Params is a tuple including h, x, and c (if LSTM)
    l = len(params)
    if l < 2:
        print('Params must be a tuple containing at least (x_t, h_t)')
        return None
    elif l > 2:
        states = (params[1], params[2])
        if rec_layer == 'lstm':
            return model.rnn(params[0].transpose(0, 1), states)
        elif rec_layer == 'gru':
            return model.gru(params[0], states)
    else:
        return model(*params)


def oneStepVarQR(J, Q):
    Z = torch.matmul(torch.transpose(J, 1, 2), Q)  # Linear extrapolation of the network in many directions
    q, r = torch.qr(Z, some=True)  # QR decomposition of new directions
    s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=1, dim2=2)))  # extract sign of each leading r value
    return torch.matmul(q, s), torch.diagonal(torch.matmul(s, r), dim1=1,
                                              dim2=2)  # return positive r values and corresponding vectors


def calc_LEs_an(*params, model, k_LE=100000, rec_layer='lstm', kappa=10, diff=10, warmup=10, T_ons=1):
    cuda = next(model.parameters()).is_cuda
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    bias = True
    cells = False  # determine whether to track cell states (for LSTM)
    x_in = Variable(params[0], requires_grad=False).to(device)
    hc = params[1]
    if len(hc) == 2:
        cells = True
        c0 = hc[1].to(device)
        h0 = Variable(hc[0], requires_grad=False).to(device)
    else:
        h0 = Variable(hc, requires_grad=False).to(device)
    num_layers, batch_size, hidden_size = h0.shape
    feed_seq, _, input_size = x_in.shape
    L = num_layers * hidden_size

    k_LE = max(min(L, k_LE), 1)
    Q = torch.reshape(torch.eye(L), (1, L, L)).repeat(batch_size, 1, 1).to(device)
    Q = Q[:, :, :k_LE]  # Choose how many exponents to track

    ht = h0
    if cells:
        ct = c0
        states = (ht, ct)
    else:
        states = (ht,)  # make tuple for easier generalization
    rvals = torch.ones(batch_size, feed_seq, k_LE).to(device)  # storage
    # qvect = torch.zeros(batch_size, feed_seq, L, k_LE) #storage
    t = 0

    # Warmup

    for xt in tqdm(x_in[:warmup], position=0, leave=True):
        xt = torch.unsqueeze(xt, 1)  # select t-th element in the fed sequence
        if cells:
            states = (ht, ct)
        else:
            states = (ht,)

        if rec_layer == 'rnn':
            J = rnn_jac(model.rnn_layer.all_weights, ht, xt, bias=bias)
        elif rec_layer == 'lstm':
            J = lstm_jac(model.rnn.all_weights, ht, ct, xt, bias=bias)
        elif rec_layer == 'gru':
            J = gru_jac(model.gru.all_weights, ht, xt, bias=bias)
        elif rec_layer == 'rhn':
            J, ht = RHN_jac(model, xt.squeeze(1), ht, bias=True)
        else:
            J = calc_Jac(xt, *states, model=model)
        if rec_layer != 'rhn':
            _, states = oneStep(xt, *states, model=model, rec_layer=rec_layer)
            if cells:
                (ht, ct) = states
            else:
                ht = states
        Q = torch.matmul(torch.transpose(J, 1, 2), Q)
        # else:
        #     Q = torch.matmul(J, Q)
    Q, _ = torch.qr(Q, some=True)

    t_QR = t
    for xt in tqdm(x_in, position=0, leave=True):
        if (t - t_QR) >= T_ons or t == 0 or t == feed_seq:
            QR = True
        else:
            QR = False
        xt = torch.unsqueeze(xt, 1)  # select t-th element in the fed sequence
        if cells:
            states = (ht, ct)
        else:
            states = (ht,)

        if rec_layer == 'rnn':
            J = rnn_jac(model.rnn_layer.all_weights, ht, xt, bias=bias)
        elif rec_layer == 'lstm':
            J = lstm_jac(model.rnn.all_weights, ht, ct, xt, bias=bias)
        elif rec_layer == 'gru':
            J = gru_jac(model.gru.all_weights, ht, xt, bias=bias)
        elif rec_layer == 'rhn':
            J, ht = RHN_jac(model, xt.squeeze(1), ht, bias=True)
        else:
            J = calc_Jac(xt, *states, model=model)

        if rec_layer != 'rhn':
            _, states = oneStep(xt, *states, model=model, rec_layer=rec_layer)
            if cells:
                ht, ct = states
            else:
                ht = states
        if QR:
            Q, r = oneStepVarQR(J, Q)
            t_QR = t
        else:
            # if rec_layer != 'rhn':
            Q = torch.matmul(torch.transpose(J, 1, 2), Q)
            # else:
            #     Q = torch.matmul(J, Q)
            r = torch.ones((batch_size, hidden_size))

        rvals[:, t, :] = r
        # qvect[:, t, :, :] = Q

        t = t + 1
    LEs = torch.sum(torch.log2(rvals.detach()), dim=1) / feed_seq
    #     print(torch.log2(rvals.detach()).shape)
    return LEs, rvals  # , qvect


def plot_evolution(rvals, k_LE, model_name='', sample_id=0, title=False, plot_size=(10, 7)):
    plt.figure(figsize=plot_size)
    feed_seq = rvals.shape[1]
    if type(k_LE == int):
        for i in range(k_LE):
            f = plt.plot(
                torch.div(torch.cumsum(torch.log2(rvals[sample_id, :, i]), dim=0), torch.arange(1., feed_seq + 1)))
    else:
        f = plt.plot(torch.div(torch.cumsum(torch.log2(rvals[sample_id, :, i]), dim=0), torch.arange(1., feed_seq + 1)))
    f = plt.xlabel('Iteration #')
    f = plt.ylabel('Lyapunov Exponent')
    if title:
        plt.title('LE Spectrum Evolution for ' + model_name + ', Sample #' + str(sample_id))
    return f


def LE_stats(LE, save_file=False, file_name='LE.p'):
    mean, std = (torch.mean(LE, dim=0), torch.std(LE, dim=0))
    if save_file:
        pkl.dump((mean, std), open(file_name, "wb"))
    return mean, std


def plot_spectrum(LE, model_name, k_LE=100000, plot_size=(10, 7), legend=[]):
    k_LE = max(min(LE.shape[1], k_LE), 1)
    LE_mean, LE_std = LE_stats(LE)
    f = plt.figure(figsize=plot_size)
    x = range(1, k_LE + 1)
    plt.title('Mean LE Spectrum for ' + model_name)
    f = plt.errorbar(x, LE_mean[:k_LE].to(torch.device('cpu')), yerr=LE_std[:k_LE].to(torch.device('cpu')), marker='.',
                     linestyle=' ', markersize=7, elinewidth=2)
    plt.xlabel('Exponent #')


def num_Jac(xt, *states, model, eps=0.01):
    if len(states) > 1:
        h, c = states
    else:
        h = states[0]
    layers, batch_size, hidden_size = h.shape
    L = layers * hidden_size
    h_flat = h.transpose(0, 1).reshape(batch_size, L, 1)
    delta = eps * torch.eye(L).repeat(batch_size, 1, 1)
    hf = h_flat.repeat(1, 1, L) + delta
    hb = h_flat.repeat(1, 1, L) - delta
    del delta
    if len(states) > 1:
        fstates = (hf, c)
        bstates = (hb, c)
    else:
        fstates = hf,
        bstates = hb,
    fwd = model.evolve_hidden(xt, *fstates)
    bwd = model.evolve_hidden(xt, *bstates)
    Jac = (fwd - bwd) / (2 * eps)
    del fwd, bwd, hf, hb, fstates, bstates
    gc.collect()
    return Jac


def lstm_jac(params_array, h, c, x, bias):
    if bias:
        W, U, b_i, b_h = param_split(params_array, bias)
    else:
        W, U = param_split(params_array, bias)
    device = get_device(h)
    num_layers, batch_size, hidden_size = h.shape
    input_shape = x.shape[-1]
    h_in = h.transpose(1, 2).detach()
    c_in = c.transpose(1, 2).detach()
    #     print(c_in)
    c_out = []
    x_in = [x.squeeze(dim=1).t()]
    if bias:
        b = [b1 + b2 for (b1, b2) in zip(b_i, b_h)]
    else:
        #         for W_i in W:
        #             print(W_i.shape)
        b = [torch.zeros(W_i.shape[0], ).to(device) for W_i in W]
    y_ones = torch.ones(hidden_size * 4, batch_size)
    y = []
    h_out = []
    i_x = slice(0 * hidden_size, 1 * hidden_size)
    f_x = slice(1 * hidden_size, 2 * hidden_size)
    c_x = slice(2 * hidden_size, 3 * hidden_size)
    o_x = slice(3 * hidden_size, 4 * hidden_size)

    J = torch.zeros(batch_size, num_layers * hidden_size, num_layers * hidden_size).to(device)
    for layer in range(num_layers):
        if layer > 0:
            x_l = tanh(c_out[layer - 1].t()).diagonal(dim1=-2, dim2=-1) * (sig(y[layer - 1])[:, o_x])
            x_in.append(x_l.t())
        y.append((W[layer] @ x_in[layer] + U[layer] @ h_in[layer] + b[layer].repeat(batch_size, 1).t()).t())
        c1 = sig(y[layer][:, f_x]).t() * c_in[layer]
        c2 = (sig(y[layer][:, i_x]) * torch.tanh(y[layer][:, c_x])).t()
        c_out.append(c1 + c2)
        h_out.append(sig(y[layer][:, o_x]).t() * torch.tanh(c_out[layer]))

        a1_h = (sigmoid_p(y[layer])[:, f_x, f_x] @ torch.diag_embed(c_in[layer].t())) @ U[layer][f_x]
        a2_h = (sigmoid_p(y[layer])[:, i_x, i_x] * tanh(y[layer])[:, c_x, c_x]) @ U[layer][i_x]
        a3_h = (sigmoid(y[layer])[:, i_x, i_x] * (sech(y[layer]) ** 2)[:, c_x, c_x]) @ U[layer][c_x]
        a_h = a1_h + a2_h + a3_h
        b_h = ((sigmoid_p(y[layer])[:, o_x, o_x]) * tanh(c_out[layer].t())) @ U[layer][o_x]
        c_h = (sigmoid(y[layer])[:, o_x, o_x] @ sech(c_out[layer].t()) ** 2)
        J_h = c_h @ a_h + b_h

        J[:, layer * hidden_size:(layer + 1) * hidden_size, layer * hidden_size:(layer + 1) * hidden_size] = J_h

        if layer > 0:
            a1_xt = (sigmoid_p(y[layer])[:, f_x, f_x] * torch.diag_embed(c_in[layer].t())) @ W[layer][f_x]
            a2_xt = (sigmoid_p(y[layer])[:, i_x, i_x] * tanh(y[layer])[:, c_x, c_x]) @ W[layer][i_x]
            a3_xt = (sigmoid(y[layer])[:, i_x, i_x] * (sech(y[layer]) ** 2)[:, c_x, c_x]) @ W[layer][c_x]
            a_xt = a1_xt + a2_xt + a3_xt
            b_xt = (sigmoid_p(y[layer])[:, o_x, o_x]) * tanh(c_out[layer].t()) @ W[layer][o_x]
            c_xt = (sigmoid(y[layer])[:, o_x, o_x] @ sech(c_out[layer].t()) ** 2)
            J_xt = c_xt @ a_xt + b_xt
            for l in range(layer, 0, -1):
                J[:, layer * hidden_size:(layer + 1) * hidden_size, (l - 1) * hidden_size:l * hidden_size] = J_xt @ J[:,
                                                                                                                    (
                                                                                                                                layer - 1) * hidden_size:(
                                                                                                                                                             layer) * hidden_size,
                                                                                                                    (
                                                                                                                                l - 1) * hidden_size:l * hidden_size]

    #     print('h_out = ' + str(h_out))
    #     print('c_out = ' + str(c_out))

    return J


def gru_jac(params_array, h, x, bias):
    if bias:
        W, U, b_i, b_h = param_split(params_array, bias)
    else:
        W, U = param_split(params_array, bias)
    device = get_device(h)
    num_layers, batch_size, hidden_size = h.shape
    input_shape = x.shape[-1]
    h_in = h.transpose(1, 2).detach()
    x_in = [x.squeeze(dim=1).t()]
    if bias:
        bi, bh = b_i, b_h
        # b = [b1 + b2 for (b1, b2) in zip(b_i, b_h)]
    else:
        bi = [torch.zeros(W_i.shape[0], ).to(device) for W_i in W]
        bh = [torch.zeros(W_i.shape[0], ).to(device) for W_i in W]
    y_ones = torch.ones(hidden_size * 3, batch_size)
    y1 = []
    y2 = []
    h_out = []
    r_x = slice(0 * hidden_size, 1 * hidden_size)
    z_x = slice(1 * hidden_size, 2 * hidden_size)
    n_x = slice(2 * hidden_size, 3 * hidden_size)
    J = torch.zeros(batch_size, num_layers * hidden_size, num_layers * hidden_size).to(device)

    h_out = []
    for layer in range(num_layers):
        if layer > 0:
            x_l = h_out[layer - 1]
            x_in.append(x_l)
        y1.append((W[layer] @ x_in[layer] + bi[layer].repeat(batch_size, 1).t()).t())
        y2.append((U[layer] @ h_in[layer] + bh[layer].repeat(batch_size, 1).t()).t())
        #         h_out.append(((1-sig(y[layer-1][:,z_x]))*sig(y[layer-1][:,n_x])).t()+torch.tanh(y[layer-1][:,z_x]).t()*h_in[layer-1])
        y = y1[layer] + y2[layer]
        n_t = torch.tanh(y1[layer][:, n_x] + sig(y[:, r_x]) * y2[layer][:, n_x])
        h_out.append(((1 - sig(y[:, z_x])) * n_t + sig(y[:, z_x]) * (h_in[layer].t())).t())
        a_h = -sigmoid_p(y[:, z_x]) @ torch.diag_embed(n_t) @ U[layer][z_x]
        b0 = torch.diag_embed(1 - sig(y[:, z_x]))
        b1 = sech(y1[layer][:, n_x] + sig(y[:, r_x]) * y2[layer][:, n_x]) ** 2
        b2_h = sigmoid_p(y[:, r_x]) @ torch.diag_embed(y2[layer][:, n_x]) @ U[layer][r_x]
        b3_h = sigmoid(y[:, r_x]) @ U[layer][n_x]
        #         print('b1 shape = {}, b2 shape = {}, b3 shape ={}'.format(b1_h.shape, b2_h.shape, b3_h.shape))
        b_h = b0 @ (b1 @ (b2_h + b3_h))
        c_h = sigmoid_p(y[:, z_x]) @ torch.diag_embed(h_in[layer].t()) @ U[layer][z_x] + sigmoid(y[:, z_x])
        J_h = a_h + b_h + c_h

        J[:, layer * hidden_size:(layer + 1) * hidden_size, layer * hidden_size:(layer + 1) * hidden_size] = J_h
        if layer > 0:
            a_xt = -sigmoid_p(y[:, z_x]) @ torch.diag_embed(n_t) @ W[layer][z_x]
            b2_x = W[layer][n_x] + sigmoid_p(y[:, r_x]) @ torch.diag_embed(y2[layer][:, n_x]) @ W[layer][r_x]
            b_xt = b0 @ (b1 @ b2_x)
            c_xt = sigmoid_p(y[:, z_x]) @ torch.diag_embed(h_in[layer].t()) @ W[layer][z_x]
            J_xt = a_xt + b_xt + c_xt
            for l in range(layer, 0, -1):
                J[:, layer * hidden_size:(layer + 1) * hidden_size, (l - 1) * hidden_size:l * hidden_size] = J_xt @ J[:,
                                                                                                                    (
                                                                                                                                layer - 1) * hidden_size:(
                                                                                                                                                             layer) * hidden_size,
                                                                                                                    (
                                                                                                                                l - 1) * hidden_size:l * hidden_size]
    return J


def rnn_jac(params_array, h, x, bias):
    if bias:
        W, U, b_i, b_h = param_split(params_array, bias)
    else:
        W, U = param_split(params_array, bias)
    device = get_device(h)
    num_layers, batch_size, hidden_size = h.shape
    input_shape = x.shape[-1]
    h_in = h.transpose(1, 2).detach()
    x_in = [x.squeeze(dim=1).t()]  # input_shape, batch_size)]
    if bias:
        b = [b1 + b2 for (b1, b2) in zip(b_i, b_h)]
    else:
        b = [torch.zeros(W_i.shape[0], ).to(device) for W_i in W]
    J = torch.zeros(batch_size, num_layers * hidden_size, num_layers * hidden_size).to(device)
    y = []
    h_out = []

    for layer in range(num_layers):
        if layer > 0:
            x_l = h_out[layer - 1]
            x_in.append(x_l)
        y.append((W[layer] @ x_in[layer] + U[layer] @ h_in[layer] + b[layer].repeat(batch_size, 1).t()).t())
        h_out.append(torch.tanh(y[layer]).t())
        J_h = sech(y[layer]) ** 2 @ U[layer]
        J[:, layer * hidden_size:(layer + 1) * hidden_size, layer * hidden_size:(layer + 1) * hidden_size] = J_h

        if layer > 0:
            J_xt = sech(y[layer]) ** 2 @ W[layer]
            for l in range(layer, 0, -1):
                J[:, layer * hidden_size:(layer + 1) * hidden_size, (l - 1) * hidden_size:l * hidden_size] = J_xt @ J[:,
                                                                                                                    (
                                                                                                                                layer - 1) * hidden_size:(
                                                                                                                                                             layer) * hidden_size,
                                                                                                                    (
                                                                                                                                l - 1) * hidden_size:l * hidden_size]
    return J


def RHN_jac(model, x, s, bias=False):
    device = get_device(s)
    if not bias:
        W, R = param_split_RHN(model, bias=bias)
    else:
        W, R, b_W, b_R = param_split_RHN(model, bias=bias)
    couple = model.couple
    s = s.squeeze(0)
    recurrence_depth = model.recurrence_depth
    num_layers = model.num_layers
    batch_size, hidden_size = s.shape
    # inputs = model.embedding(x)
    input_shape = x.shape[-1]

    if couple:
        HTC_indices = [hidden_size, hidden_size]
    else:
        HTC_indices = [hidden_size, hidden_size, hidden_size]
    J = torch.eye(hidden_size).repeat(batch_size, 1, 1).to(device)
    # J = torch.zeros(batch_size, hidden_size, hidden_size).to(device)
    for i, layer in enumerate(range(num_layers)):
        # W[layer] = W[layer].to(torch.long)
        if not couple:
            W_H, W_T, W_C = torch.split(W[layer], HTC_indices)
        else:
            W_H, W_T = torch.split(W[layer], HTC_indices)
    # print(W_H.shape, W_T.shape, W_C   .shape)
        if layer == 0:
            x = x.unsqueeze(2).to(torch.float32)
        else:
            x = s.unsqueeze(2)
        for j, rec in enumerate(range(recurrence_depth)):
            s = s.unsqueeze(2)
            if rec == 0:
                if not couple:
                    R_H, R_T, R_C = torch.split(R[layer][rec], HTC_indices)
                    h = torch.tanh(torch.matmul(W_H, x) + torch.matmul(R_H, s)).squeeze(2)
                    t = torch.sigmoid(torch.matmul(W_T, x) + torch.matmul(R_T, s)).squeeze(2)
                    c = torch.sigmoid(torch.matmul(W_C, x) + torch.matmul(R_C, s)).squeeze(2)

                    H = torch.matmul(R_H.transpose(0, 1), torch.diag_embed(1 - h**2))
                    T = torch.matmul(R_T.transpose(0, 1), torch.diag_embed(t * (1-t)))
                    C = torch.matmul(R_C.transpose(0, 1), torch.diag_embed(c * (1-c)))
                    J_t = torch.matmul(H, torch.diag_embed(t)) + torch.matmul(T, torch.diag_embed(h)) + \
                        torch.matmul(C, torch.diag_embed(s.squeeze(2))) + torch.diag_embed(c)
                else:
                    R_H, R_T, = torch.split(R[layer][rec], HTC_indices)
                    b_H, b_T, = torch.split(b_R[layer][rec], HTC_indices)
                    h = torch.tanh((torch.matmul(W_H, x) + torch.matmul(R_H, s)).squeeze(2) + b_H)
                    t = torch.sigmoid((torch.matmul(W_T, x) + torch.matmul(R_T, s)).squeeze(2) + b_T)
                    c = 1 - t
                    # c = torch.sigmoid(torch.matmul(W_C, x) + torch.matmul(R_C, s)).squeeze(2)

                    H = torch.matmul(R_H.transpose(0, 1), torch.diag_embed(1 - h ** 2))
                    T = torch.matmul(R_T.transpose(0, 1), torch.diag_embed(t * (1 - t)))
                    # C = torch.matmul(R_C.transpose(0, 1), torch.diag_embed(c * (1 - c)))
                    J_t = torch.matmul(H, torch.diag_embed(t)) + torch.matmul(T, torch.diag_embed(h)) + \
                          torch.eye(hidden_size).to(device) - (torch.diag_embed(t) + torch.matmul(T, torch.diag_embed(s.squeeze(2))))
                    # J = J_t
            else:
                if not couple:
                    R_H, R_T, R_C = torch.split(R[layer][rec], HTC_indices)
                    h = torch.tanh(torch.matmul(R_H, s).squeeze(2))
                    t = torch.sigmoid(torch.matmul(R_T, s)).squeeze(2)
                    c = torch.sigmoid(torch.matmul(R_C, s)).squeeze(2)

                    H = torch.matmul(R_H.transpose(0, 1), torch.diag_embed(1 - h ** 2))
                    T = torch.matmul(R_T.transpose(0, 1), torch.diag_embed(t * (1 - t)))
                    C = torch.matmul(R_C.transpose(0, 1), torch.diag_embed(c * (1 - c)))
                    J_t = torch.matmul(H, torch.diag_embed(t)) + torch.matmul(T, torch.diag_embed(h)) + \
                        torch.matmul(C, torch.diag_embed(s.squeeze(2))) + torch.diag_embed(c)
                else:
                    R_H, R_T = torch.split(R[layer][rec], HTC_indices)
                    b_H, b_T, = torch.split(b_R[layer][rec], HTC_indices)
                    h = torch.tanh(torch.matmul(R_H, s).squeeze(2) + b_H)
                    t = torch.sigmoid(torch.matmul(R_T, s).squeeze(2) + b_T)
                    c = 1 - t

                    H = torch.matmul(R_H.transpose(0, 1), torch.diag_embed(1 - h ** 2))
                    T = torch.matmul(R_T.transpose(0, 1), torch.diag_embed(t * (1 - t)))
                    J_t = torch.matmul(H, torch.diag_embed(t)) + torch.matmul(T, torch.diag_embed(h)) + torch.eye(hidden_size).to(device) - (torch.diag_embed(t) + torch.matmul(T, torch.diag_embed(s.squeeze(2))))
            J = torch.matmul(J, J_t)
            # J *= J_t
            s = h * t + s.squeeze(2) * c
    return J, s.unsqueeze(0)

def param_split(model_params, bias):
    #   model_params should be tuple of the form (W_i, W_h, b_i, b_h)
    hidden_size = int(model_params[0][0].shape[0])
    layers = len(model_params)
    W = []
    U = []
    b_i = []
    b_h = []
    if bias:
        param_list = (W, U, b_i, b_h)
    else:
        param_list = (W, U)
    grouped = []
    for idx, param in enumerate(param_list):
        for layer in range(layers):
            #             if len(param.shape) == 1:
            #                 param = param.squeeze(dim=1)
            param.append(model_params[layer][idx].detach())
        grouped.append(param)
    return grouped

def param_split_RHN(model, bias=True):
    W = []
    R = []
    weights = []
    if bias:
        W_bias = []
        R_bias =[]
    for i, (name, param) in enumerate(model.named_parameters()):
        # print(i, name)
        if 'rnns' in name:
            weights.append(param.data)
            layer = int(name.split('.')[1])
            recurrent = int(name.split('.')[3])
            if 'W.weight' in name:
                W.append(param.data)
            elif 'W_C.weight' in name:
                W[layer] = torch.cat((W[layer], param.data), dim=0)
            elif 'W.bias' in name:
                W_bias.append(param.data)
            elif 'W_C.bias' in name:
                W_bias[layer] = torch.cat((W_bias[layer], param.data), dim=0)

            if 'R.weight' in name:
                if recurrent == 0:
                    R.append([param.data])
                else:
                    R[layer].append(param.data)
            elif 'R_C.weight' in name:
                R[layer][recurrent] = torch.cat((R[layer][recurrent], param.data), dim=0)
            elif 'R.bias' in name:
                if recurrent == 0:
                    R_bias.append([param.data])
                else:
                    R_bias[layer].append(param.data)
            elif 'R_C.bias' in name:
                R_bias[layer][recurrent] = torch.cat((R_bias[layer][recurrent], param.data), dim=0)
    if bias:
        return W, R, W_bias, R_bias
    else:
        return W, R


## Define Math Functions
def get_device(X):
    if X.is_cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def sig(X):
    device = get_device(X)
    return 1 / (1 + torch.exp(-X))


def sigmoid(X):
    device = get_device(X)
    return torch.diag_embed(1 / (1 + torch.exp(-X)))


def sigmoid_p(X):
    device = get_device(X)
    ones = torch.ones_like(X)
    return torch.diag_embed(sig(X) * (ones - sig(X)))


def sech(X):
    device = get_device(X)
    return torch.diag_embed(1 / (torch.cosh(X)))


def tanh(X):
    device = get_device(X)
    return torch.diag_embed(torch.tanh(X))

def main():
    experiment_packet = pickle.load(open('experiment_packet.pickle', 'rb'))
    # print(experiment_packet)
    model = experiment_packet['model']
    x = experiment_packet['x']
    h = torch.zeros(model.num_layers, x.size(0), model.hidden_size).to(model.device)
    c = torch.zeros(model.num_layers, x.size(0), model.hidden_size).to(model.device)
    params = (x, (h, c))
    # print(h0.shape)
    # bias = True
    # params_array = model.lstm.all_weights
    # J = lstm_jac(params_array, h, c, x, bias)

    LEs, rvals = calc_LEs_an(*params, model=model)
    print(LEs, rvals)
if __name__ == "__main__":
    main()