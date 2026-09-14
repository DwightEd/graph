"""Run the source-disjoint structural attention/entropy detector experiment."""
import runpy

if __name__ == '__main__':
    runpy.run_module('structural_detector.experiment',run_name='__main__')
