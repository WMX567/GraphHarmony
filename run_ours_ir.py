import argparse
from model_block.data_loader import *
from model_block.model_imdb_reddit import *
from model_block.optimize import *
from model_block.utils import *
from model_block.earlystopping import EarlyStoppingF1
from model_block.nwd import NuclearWassersteinDiscrepancy
import gc

def evaluate(model, loader, domain, device):
    model.eval()
    y_true, _, y_score = [], [], []
    with torch.no_grad():
        for batch in loader:
            adj, feats, labels, vertices = [tmp.to(device) for tmp in batch]
            adj = add_self_loop(adj)
            out = model(feats, vertices, adj, domain)
            y_true += labels.data.tolist()
            y_score += out['cls_output'][:, 1].data.tolist()
    auc, f1 = get_metrics(y_true, y_score)
    print('Eval AUC={:.4f} F1={:.4f}'.format(auc, f1))
    return auc, f1


def run(args):
  
    src_ds, n_feat, src_class_weight, src_train_loader,_, _ = load_IMDB_and_REDDIT_dataset(
        path=args.data_path + args.src_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        num_workers=args.num_workers)
    
    device = torch.device(args.device)
    src_class_weight = torch.FloatTensor(src_class_weight).to(device)
    beta = torch.FloatTensor([args.beta]).to(device)
    ent_weight = torch.FloatTensor([args.ent_weight]).to(device)
    y_w = torch.FloatTensor([args.y_weight]).to(device)
    m_w = torch.FloatTensor([args.m_weight]).to(device)
    weights = [src_class_weight, beta, ent_weight, y_w, m_w]

    gc.collect()
    tar_ds, _, _, tar_train_loader, tar_val_loader, _ = load_IMDB_and_REDDIT_dataset(
        path=args.data_path + args.tar_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        num_workers=args.num_workers)
    
    seed = 27 + args.r
    print('Seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model_path = 'OurBase_{}2{}_{}_{}.pth'.format(args.src_data, args.tar_data, args.backbone, str(args.r))
    # model definition
    model = Our_Base_noise(n_feat, args.enc_hidden_dim, args.m_dim, args.droprate,
                src_ds.get_vertex_features(),
                tar_ds.get_vertex_features(), args.backbone)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
    discrepancy = NuclearWassersteinDiscrepancy(model.classClassifier)

    n_batch = max(len(src_train_loader), len(tar_train_loader))
    s_iter = iter(src_train_loader)
    t_iter = iter(tar_train_loader)
    mn_loss = tr_loss = best_val_f1 = 0.0

    # Train
    early_stopping = EarlyStoppingF1(patience=args.patience, verbose=True, save_path=model_path)
    for batch_idx in range(args.epoch * n_batch):
        # reset batch iterator
        if batch_idx % len(src_train_loader) == 0:
            s_iter = iter(src_train_loader)
        if batch_idx % len(tar_train_loader) == 0:
            t_iter = iter(tar_train_loader)

        # load batch data
        s_adj, s_feats, s_labels, s_vts = next(s_iter)
        t_adj, t_feats, t_labels, t_vts = next(t_iter)

        s_adj = add_self_loop(s_adj)
        t_adj = add_self_loop(t_adj)

        s_adj = s_adj.to(device)
        s_feats = s_feats.to(device)
        s_labels = s_labels.to(device)
        s_vts = s_vts.to(device)
        t_adj = t_adj.to(device)
        t_feats = t_feats.to(device)
        t_labels = t_labels.to(device)
        t_vts = t_vts.to(device)

        # train with original data
        s_out = model(s_feats, s_vts, s_adj, 0)
        t_out = model(t_feats, t_vts, t_adj, 1)
        src_tr_loss = Our_New_loss(s_out, s_labels, s_adj.clone(), 0, weights)
        tar_tr_loss = Our_New_loss(t_out, None, t_adj.clone(), 1, weights, target_loss_w=0.001)
        x = torch.cat((s_out['emb'], t_out['emb']), dim=0)
        transfer_loss = -discrepancy(x)
        tr_loss += src_tr_loss + tar_tr_loss + transfer_loss

        # train with manipulated data
        s_nadj, _ = drop_edges(s_adj, args.edge_drop_rate, args.edge_add_rate)
        t_nadj, _ = drop_edges(t_adj, args.edge_drop_rate, args.edge_add_rate)
        s_out = model(s_feats, s_vts, s_nadj, 0)
        t_out = model(t_feats, t_vts, t_nadj, 1)
        s_mn_loss = Our_New_loss(s_out, s_labels, s_adj.clone(), 0, weights)
        t_mn_loss = Our_New_loss(t_out, t_labels, t_adj.clone(), 1, weights)
        x = torch.cat((s_out['emb'], t_out['emb']), dim=0)
        aug_transfer_loss = -discrepancy(x)
        mn_loss += s_mn_loss + t_mn_loss + aug_transfer_loss

        if (batch_idx + 1) % args.real_batch_size == 0:
            tr_loss /= args.real_batch_size
            tr_loss.backward()
            mn_loss /= args.real_batch_size
            mn_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            lr_scheduler.step()
            mn_loss = tr_loss = 0

        # Validate
        if (batch_idx + 1) % args.real_batch_size == 0:
            _, f1 = evaluate(model, tar_val_loader, 1, device)
            if best_val_f1 < f1:
                best_val_f1 = f1
            early_stopping(f1, model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        
    print("Finish!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OurBase')
    parser.add_argument('--data_path', default='infodata/data/')
    parser.add_argument('--src_data', default='twitter',
                        help='Source domain dataset. (oag, twitter, weibo, digg)')
    parser.add_argument('--tar_data', default='oag')
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--backbone', default='gat', help='Backbone Feature Extractor GNN. gcn / gat / gin')
    parser.add_argument('--enc_hidden_dim', type=int, default=128,
                        help='Dimension of the feature extractor hidden layer. Default is 256. ')
    parser.add_argument('--d_dim', type=int, default=256,
                        help='Dimension of the domain latent variables. Default is 64. ')
    parser.add_argument('--y_dim', type=int, default=128,
                        help='Dimension of the semantic latent variables. Default is 256. ')
    parser.add_argument('--m_dim', type=int, default=128,
                        help='Dimension of the semantic latent variables. Default is 256. ')
    parser.add_argument('--dec_hidden_dim', type=int, default=64,
                        help='Dimension of the graph decoder hidden layer. Default is 64. ')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--real_batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.75)
    parser.add_argument('--val_ratio', type=float, default=0.125)
    parser.add_argument('--recons_weight', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--ent_weight', type=float, default=1)
    parser.add_argument('--m_weight', type=float, default=0.1)
    parser.add_argument('--d_weight', type=float, default=1)
    parser.add_argument('--y_weight', type=float, default=1)
    parser.add_argument('--check_batch', type=float, default=1)
    parser.add_argument('--validate_batch', type=float, default=1)
    parser.add_argument('--lr_decay_epoch', type=int, default=30)
    parser.add_argument('--manipulate_batch', type=int, default=1)
    parser.add_argument('--edge_add_rate', type=float, default=0.1)
    parser.add_argument('--edge_drop_rate', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()
    
    run(args)
