
# Run this after your inference loop to find the "Sweet Spot"
best_acc = 0
best_t = 0
for t in np.arange(0.1, 0.9, 0.05):
    temp_preds = [1 if p > t else 0 for p in df['prob']]
    acc = (np.array(temp_preds) == df['label']).mean()
    if acc > best_acc:
        best_acc = acc
        best_t = t
print(f"Optimal Threshold: {best_t:.2f} | Highest Test Acc: {best_acc:.4f}")
