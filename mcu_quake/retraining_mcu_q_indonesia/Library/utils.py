# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import pandas as pd
import config
import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl


def plot_training(H, output_path):
    def chart_lines(ax, line1, line2, line1_label, line2_label, x_label, y_label):
        ax.plot(line1, label=line1_label)
        ax.plot(line2, label=line2_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="lower left")        

    # Build dataframe
    history_data = pd.DataFrame({
        "Train loss": H.history["loss"],
        "Train acc": H.history["acc"],
        "Validation loss": H.history["val_loss"],
        "Validation acc": H.history["val_acc"],
    })
    data_path = os.path.sep.join([output_path, f"Train history, {config.FEATURE_DISTANCE}.csv"])
    history_data.to_csv(data_path, index=False)

    # Plot figure
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Training metrics')

    # Plot loss
    chart_lines(ax=ax1,
                line1=history_data["Train loss"],
                line2=history_data["Validation loss"],
                line1_label="Train",
                line2_label="Validation",
                x_label="Epoch", y_label="Loss")
        
    # Plot accuracy
    chart_lines(ax=ax2,
                line1=history_data["Train acc"],
                line2=history_data["Validation acc"],
                line1_label="Train",
                line2_label="Validation",
                x_label="Epoch", y_label="Accuracy")

    fig.savefig(os.path.sep.join([output_path, f"Plot train, {config.FEATURE_DISTANCE}.jpg"]))


def embedding_PDFs_1D(embedding_X, source_list=["noise", "le"]):
    """
    embedding_X: {"noise": data, "le": data}
    """
    PDF_Normals = {}
    PDF_KDEs = {}

    for source_label in source_list:
        if source_label in embedding_X:
            values = np.array(embedding_X[source_label])
            if len(values) > 0:
                # Normal distribution
                PDF_Normals[source_label] = stats.norm(np.mean(values), np.std(values)) 
                # Kernel density estimation
                PDF_KDEs[source_label] = stats.gaussian_kde(values.T)                    

    return {"Norm": PDF_Normals, "Kernel": PDF_KDEs}


def embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E, source_list=["noise", "le"]):
    """
    embedding_Z/N/E: {"noise": data, "le": data}
    """   
    PDF_Normals = {}
    PDF_KDEs = {}

    for source_label in source_list:
        if source_label in embedding_Z and source_label in embedding_N and source_label in embedding_E:
            values_Z = np.array(embedding_Z[source_label])
            values_N = np.array(embedding_N[source_label])
            values_E = np.array(embedding_E[source_label])

            if len(values_Z) > 0:
                samples_3D = np.column_stack((values_E, values_N, values_Z))

                # Normal distribution
                means_estimated = np.mean(samples_3D, axis=0)
                cov_matrix_estimated = np.cov(samples_3D.T)
                PDF_Normals[source_label] = stats.multivariate_normal(mean=means_estimated, cov=cov_matrix_estimated)

                # Kernel density estimation
                PDF_KDEs[source_label] = stats.gaussian_kde(samples_3D.T)     
    
    return {"Norm": PDF_Normals, "Kernel": PDF_KDEs}


def embedding_likelihood(input_embedding, embeddings_PDF):
    likelihood_noise = embeddings_PDF["noise"].pdf(input_embedding)
    likelihood_le = embeddings_PDF["le"].pdf(input_embedding)

    return {
        "noise": float(likelihood_noise),
        "le": float(likelihood_le)
    }


def infer_1C_PDFs(input_embedding, embeddings_PDFs, choose_pdf="Kernel"):
    """
    0: noise
    1: le
    """
    likelihood_dict = embedding_likelihood(input_embedding, embeddings_PDFs[choose_pdf])

    # Softmax biner (noise vs le)
    likelihood_softmax = softmax([likelihood_dict["noise"], likelihood_dict["le"]])
    
    # 0 -> noise, 1 -> le
    infer_type = np.argmax(likelihood_softmax)

    return int(infer_type), likelihood_dict, list(np.round(likelihood_softmax, 4))


def infer_3C_PDFs(input_embeddings_3C, embeddings_3C_PDFs, choose_pdf="Kernel"):
    likelihood_dict = embedding_likelihood(input_embeddings_3C, embeddings_3C_PDFs[choose_pdf])

    # Softmax biner (noise vs le)
    likelihood_softmax = softmax([likelihood_dict["noise"], likelihood_dict["le"]])
    
    infer_type = np.argmax(likelihood_softmax)

    return int(infer_type), likelihood_dict, list(np.round(likelihood_softmax, 4))


def two_class_convert(true_list, pred_list):
    """
    Konversi proteksi jika ada label > 1 menjadi 1 (seismic)
    0: noise
    1: seismic / le
    """
    two_class_true = list(np.where(np.array(true_list) > 1, 1, np.array(true_list)))
    two_class_pred = list(np.where(np.array(pred_list) > 1, 1, np.array(pred_list)))

    return two_class_true, two_class_pred


def calc_confusion_metrics(true_labels, predicted_labels):
    cnf_matrix = confusion_matrix(true_labels, predicted_labels)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    ACC = (TP + TN) / (TP + FP + FN + TN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP) 
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)
    F1 = 2 * (PPV * TPR) / (PPV + TPR)

    # Weighted Averages
    total = cnf_matrix.sum()
    row_sum = cnf_matrix.sum(axis=1)

    metrics = {
        "Accuracy": list(ACC), "Accuracy (avg.)": np.sum(row_sum * ACC) / total,
        "True positive rate": list(TPR), "True positive rate (avg.)": np.sum(row_sum * TPR) / total,
        "True negative rate": list(TNR), "True negative rate (avg.)": np.sum(row_sum * TNR) / total,
        "Positive predictive value": list(PPV), "Positive predictive value (avg.)": np.sum(row_sum * PPV) / total,
        "Negative predictive value": list(NPV), "Negative predictive value (avg.)": np.sum(row_sum * NPV) / total,
        "False positive rate": list(FPR), "False positive rate (avg.)": np.sum(row_sum * FPR) / total,
        "False negative rate": list(FNR), "False negative rate (avg.)": np.sum(row_sum * FNR) / total,
        "False discovery rate": list(FDR), "False discovery rate (avg.)": np.sum(row_sum * FDR) / total,
        "F1-score": list(F1), "F1-score (avg.)": np.sum(row_sum * F1) / total,
    }

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return cnf_matrix, metrics


def latent_codes_1D(data, model):
    _input = np.array(data).reshape(1, -1, 1)
    _embedding = model(_input).numpy()[0].reshape(1, -1)
    return _embedding[0]


def plot_confusion(title, true_labels, matrix, metrics=None,
                   fig_title="Confusion matrix", axLabelSize=8, axTickSize=8,
                   fig_size=[5, 3.9], subAdjust=(0.18, 0.82, 0.2, 0.8)):

    mpl.rcParams['axes.titlesize'] = axLabelSize    
    mpl.rcParams['xtick.labelsize'] = axTickSize  
    mpl.rcParams['ytick.labelsize'] = axTickSize
    mpl.rcParams['axes.labelsize'] = axTickSize    
    mpl.rcParams['axes.labelpad'] = 2               
    mpl.rcParams['legend.fontsize'] = axTickSize   
    mpl.rcParams['savefig.dpi'] = 300 

    cm = 1 / 2.54
    fig = plt.figure(figsize=(fig_size[0] * cm, fig_size[1] * cm), dpi=165)
    fig.subplots_adjust(left=subAdjust[0], right=subAdjust[1], bottom=subAdjust[2], top=subAdjust[3])  

    fig.canvas.manager.set_window_title(fig_title)
    ax = fig.add_subplot(111)

    ax.set_title(title, size=axLabelSize)
    ax.set_ylabel('Ground truth')

    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=true_labels).plot(
        include_values=True, ax=ax, colorbar=False, values_format="",
        text_kw={"fontsize": axTickSize}, cmap="Blues"
    )
    
    if metrics is None:
        x_ticklabels = ['' for _ in range(len(matrix))]
    else:
        x_ticklabels = np.round(np.array(metrics["Positive predictive value"]) * 100, 2)
        ax.annotate('PPV:', xy=(0, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(-33, -13), textcoords="offset points", fontsize=axTickSize)
        
        y_ticklabels = np.round(np.array(metrics["True positive rate"]) * 100, 2)
        num_tick = len(matrix) * 2 + 1
        vals = np.linspace(-0.5, len(matrix) - 0.5, num_tick)
        tick_value = vals[1:-1:2] + axTickSize * 0.005

        for val, lab in zip(tick_value, y_ticklabels):
            ax.text(1.1, val, lab, transform=ax.get_yaxis_transform(), fontsize=axTickSize)
            
        ax.text(1.1, -0.55, "TPR:", transform=ax.get_yaxis_transform(), fontsize=axTickSize)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.xaxis.set_ticklabels(x_ticklabels)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted')
    ax.set_aspect(aspect='equal')

    return fig