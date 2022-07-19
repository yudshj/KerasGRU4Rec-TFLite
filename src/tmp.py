        recall = []
        mrr = []
        loss = []
        for feat, mask, target in tqdm(zip(feats, masks, targets)):
            result = m.eval(feat, target, last_state, top_k=20)
            recall.append(result['recall'].numpy())
            mrr.append(result['mrr'].numpy())
            loss.append(result['loss'].numpy())
            last_state = result['state']
        recall = np.mean(recall)
        mrr = np.mean(mrr)
        loss = np.mean(loss)
        print(f"[ EVALUATE {epoch} ] recall: {recall:.5f}, mrr: {mrr:.5f}, loss: {loss:.5f}")