import argparse
import shlex

def parser_args_to_cmdline(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    parts = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.help == argparse.SUPPRESS:
            continue
        val = getattr(args, action.dest, None)
        if val is None:
            continue
        opt = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if val:
                parts.append(opt)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not val:
                parts.append(opt)
            continue
        if isinstance(val, (list, tuple)):
            for v in val:
                parts.append(opt)
                parts.append(str(v))
        else:
            parts.append(opt)
            parts.append(str(val))
    return ' '.join(shlex.quote(p) for p in parts)


def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--ModelName', default='PathoGenCVFM', type=str)
	parser.add_argument('--dataset', default='test', type=str, help='test | hprd | iid | string')
	parser.add_argument('--SplitMode', default='disMid', type=str, help='disMid | dis')
	parser.add_argument('--data_source', default='[1,2,3,4,5,6,7]', type=str, help='ids of data sources: 1:gene_gene","2:gene_go","3:gene_dis","4:dis_gene","5:dis_dis","6:dis_do","7:dis_hpo')
	parser.add_argument('--Ks', default='[1,5,10,15,20,25,30,35,50,100,150,200,250,300]', type=str, help='topK')
	parser.add_argument('--mode', default='train_test', type=str, help='train_test | infer | finetuning | casestudy | foldcv')
	parser.add_argument('--Exp_name', default='Exp_name', type=str, help='Experiment Name, as well as the secondary path for save and inference')
	parser.add_argument('--tgtDis', default="['C0002395']", type=str, help='diseaseID list for CaseStudy')

	parser.add_argument('--epoch', default=10, type=int, help='number of epochs')
	parser.add_argument('--batch', default=128, type=int, help='training batchsize')
	parser.add_argument('--tstBat', default=128, type=int, help='testing batchsize')
	parser.add_argument('--isbalance', action='store_false',default=True)
	parser.add_argument('--neg_times_case', type=int, default=1, help='The multiple of negative samples in CaseStudy')
	parser.add_argument('--n_fold', type=int, default=5, help='n in n-fold cross validation')
	parser.add_argument('--exe_fold', type=int, default=-1, help='execute fold')

	parser.add_argument("--seed", type=int, default=2025, help="random seed")
	parser.add_argument('--gpu', default='0', type=str, help='gpu id')

	parser.add_argument('--View_emb_dim', default=128, type=int, help='MetaHGT embedding size')
	parser.add_argument('--n_heads', type=int, default=1, help='n_heads in self-attention')
	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')

	parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
	parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')
	parser.add_argument('--reg', default=0.001, type=float, help='weight decay regularizer (lambda2)') # 1e-4
	parser.add_argument('--denoise_emb_size', type=int, default=10)
	parser.add_argument('--norm', action='store_true',default=False)
	parser.add_argument('--steps', type=int, default=3, help='timesteps of diffusion (T)')
	parser.add_argument('--noise_scale', type=float, default=1.0)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.02)
	parser.add_argument('--sampling_noise', action='store_true',default=False)
	parser.add_argument('--sampling_steps', type=int, default=0)
	parser.add_argument('--rebuild_k_percent', type=float, default=0.003)  # 0.0015
	parser.add_argument('--lambda_0', type=float, default=0.05, help='lambda0')  # 0.1
	parser.add_argument('--ris_lambda', type=float, default=0.5, help='w (in M_Main)')
	parser.add_argument('--ris_adj_lambda', type=float, default=0.2)  # 0.2
	parser.add_argument('--View_transType', type=str, default='MLP', help='MLP or Random')

	parser.add_argument('--latdim', default=512, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
	parser.add_argument('--data', default='allrecipes', type=str, help='name of dataset')
	parser.add_argument('--cl_loss', default=0.01, type=float, help='weight for contrative learning (lambda1)') # 1e-2
	parser.add_argument('--tau', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--cl_method', type=str, default='Main_Sub', help='Main_Sub or Sub_Sub')

	return parser, parser.parse_args()

parser, parserargs = ParseArgs()
args = vars(parserargs)
cmd_str = parser_args_to_cmdline(parser, parser.parse_args([]))


