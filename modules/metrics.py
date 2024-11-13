from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sksurv.metrics import concordance_index_censored

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def compute_cindex(samples):
    all_event_times=[]
    all_estimate=[]
    for case in samples.keys(): 
        qca = samples[case]
        for item in qca:
            if 'survival time' in item['Question'][0]:
                res = item['res']
                gts = item['gts']

                if not res.isdecimal():
                    continue
                all_event_times.append(eval(gts))
                all_estimate.append(eval(res))

    return concordance_index_censored([True]*len(all_estimate), all_event_times, all_estimate, tied_tol=1e-08)[0]
