import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        # args for path
        parser.add_argument('--raw_data_dir', default='./data',
                            help='the data dir of raw data')

        parser.add_argument('--output_dir', default='./out/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='../bert_model_data/pre_bert/torch_roberta_wwm',
                            help='bert dir for ernie / roberta-wwm / uer')

        parser.add_argument('--task_type', default='span',
                            help='crf / span / mrc')

        parser.add_argument('--loss_type', default='ls_ce',
                            help='loss type for span/mrc')
        # other args
        parser.add_argument('--seed', type=int, default=2022, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default=['0'],
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--mode', type=str, default='train',
                            help='train / stack')
        # train args
        parser.add_argument('--train_epochs', default=5, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate for the bert module')

        parser.add_argument('--other_lr', default=2e-3, type=float,
                            help='learning rate for the module except bert')

        parser.add_argument('--max_grad_norm', default=1.0, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=64, type=int)

        parser.add_argument('--attack_train', default='fgm', type=str,
                            help='fgm / pgd attack train when training')
        parser.add_argument('--test_file', default='')


        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
