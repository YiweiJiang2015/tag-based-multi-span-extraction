import sys
import numpy as np
from collections import defaultdict

# Get the `ScalarMix` parameters from a trained model. These parameters
# correspond to the learned scalar weights for each encoder layer in BERT.


def main(args):

    from allennlp.common.util import import_submodules
    # --include-package
    import_submodules(args[1])

    from allennlp.models.archival import load_archive
    # File name of the model archive from which we will extract the weights.
    archive = load_archive(args[0], weights_file=args[2])

    heads_to_weights = defaultdict(lambda: {})
    for n, p in archive.model.named_parameters():
        if '.scalar_parameters.' in n:
            head = n.partition('.')[0].replace('_scalar_mix_', '')
            layer = int(n.rpartition('.')[-1])
            heads_to_weights[head][layer] = p.data[0]

    for head, weights_dict in heads_to_weights.items():
        params = np.zeros(len(weights_dict))
        for k, v in weights_dict.items():
            params[k] = v 
        heads_to_weights[head] = np.exp(params) / np.sum(np.exp(params))
        print(head)
        print(heads_to_weights[head])
        print()

if __name__ == "__main__":
    main(sys.argv[1:])