"""Detection entry. `transport` runs S11; old flags keep the frozen S10 interface."""
import sys

if __name__ == '__main__':
    if sys.argv[1:2] == ['transport']:
        from structural_detector.transport_run import main
        main(sys.argv[2:])
    else:
        import runpy
        runpy.run_module('structural_detector.experiment', run_name='__main__')
