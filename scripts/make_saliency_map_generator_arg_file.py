import os


def main():
    checkpoint_paths = []
    model_aliases = []
    splits = ['validation_mc', 'test']

    trials_path = '../output/trained_models/vggdrop_norm_abn_final'
    for trial_dir in os.listdir(trials_path):
        trial_path = os.path.join(trials_path, trial_dir)
        for subdir in os.listdir(trial_path):
            if 'checkpoint' in subdir:
                checkpoint_path = os.path.join(trial_path, subdir)
        model_aliases.append(trial_dir)
        checkpoint_paths.append(checkpoint_path)  


    arg_file = './arg_files/saliency_map_generator_20_trials_args.txt'
    with open(arg_file, 'w') as f:
        for checkpoint_path, alias in zip(checkpoint_paths, model_aliases):
            for split in splits:
                f.write(f"{checkpoint_path} {alias} {split}\n")

if __name__ == '__main__':
    main()

