import hydra


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    print(args)
    print(args.data.filename.get('train'))


if __name__ == "__main__":
    main()
