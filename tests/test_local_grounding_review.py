import unittest
from next_iteration.local_grounding_review import messages, add_audit, audit_messages


class ReviewTests(unittest.TestCase):
    def test_review_is_not_ground_truth(self):
        original=messages('source','task','answer',(0,6))
        updated=add_audit(original,'fallible draft',True)
        self.assertNotIn('DRAFT_AUDIT',original[1]['content'])
        self.assertIn('not evidence or a label',updated[1]['content'])
        self.assertIn('truncated',updated[1]['content'])
        self.assertLess(updated[1]['content'].index('<DRAFT_AUDIT>'),updated[1]['content'].index('Target character'))

    def test_audit_only_source_evidence(self):
        result=audit_messages('raw source','task','original answer')
        self.assertIn('Use only SOURCE',result[0]['content'])
        self.assertIn('original answer',result[1]['content'])
        self.assertIn('raw source',result[1]['content'])

    def test_no_draft_masquerades_as_complete(self):
        result=add_audit(messages('s','t','a',(0,1)),'x',False)
        self.assertIn('not guaranteed complete',result[1]['content'])


if __name__=='__main__':unittest.main()
