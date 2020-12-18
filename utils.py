import numpy as np
import ml_metrics

def from_probs_to_products(probs, products=["A-A", "B-B", "C-D", "D-E", "E-E"]):
  return [list(np.array(products)[np.argsort(prob)[::-1].astype(int)]) for prob in probs]

# Evaluation
def from_preds_to_products(preds, products=["A-A", "B-B", "C-D", "D-E", "E-E"]):
  if np.array(preds).shape[-1] == 2:
    preds = np.array([pred[:,1] for pred in preds]).transpose()
  product_preds = from_probs_to_products(preds)
  return product_preds

def mAP(y_true, y_pred, products=["A-A", "B-B", "C-D", "D-E", "E-E"]):
  product_preds = from_preds_to_products(preds=y_pred, products=products)
  actuals = [list(np.array(products)[x==1]) for x in y_true]
  return mAP_(actual=actuals, predicted=product_preds)

def mAP_(actual, predicted):
  AP = [ml_metrics.apk(actual=actual[i], predicted=predicted[i], k=5) if len(actual[i]) else 0 for i in range(len(predicted)) ]
  mAP = np.mean(AP)
  std = np.std(AP)/np.sqrt(len(AP))
  #print(f'Error: {mAP:.4f} +/- {3*std:.4f}')
  return mAP