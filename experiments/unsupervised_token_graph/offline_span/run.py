"""旧来源错接训练已退出主线，禁止旧命令继续启动训练。"""


def main(argv=None):
    raise SystemExit(
        'offline_span training is retired. Use: '
        'python -m experiments.unsupervised_token_graph.span_audit --help\n'
        'Current task: labelled mechanism comparisons; see iclr/MECHANISM_FIRST.md.'
    )


if __name__ == '__main__':
    main()
