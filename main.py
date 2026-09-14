"""Default is label-free local reuse; historical supervised baselines are explicit."""
import sys

if __name__ == '__main__':
    command = sys.argv[1:2]
    if command == ['transport']:
        raise SystemExit('S11 transport is supervised. Use supervised-s11 explicitly; default main.py is unsupervised.')
    if command == ['supervised-s11']:
        from structural_detector.transport_run import main
        main(sys.argv[2:])
    elif command == ['supervised-s10']:
        import runpy
        del sys.argv[1]
        runpy.run_module('structural_detector.experiment', run_name='__main__')
    else:
        if command == ['unsupervised']:
            del sys.argv[1]
        from reuse_detector.run import main
        main()
