def _plot_batches(embedding=False, concat=True):
    """
    Plot the batches in a T-SNE plot.

    embedding: bool
        If True, it will plot the full embedding of the batches. Else, will just plot fp of reactants.
    concat: bool
        If True, it will plot the fp of reactants concatenated. Else, will plot them separately.
    """
    filename = '/Users/kate_fieseler/PycharmProjects/edboplus_scope/test/thompson_SNAr/figure_S2/benchmark_init_random_embedding_OHE_scaling_log_batch_5_acq_EHVI.csv'

    df_preds = pd.read_csv(filename)
    r1_test = df_preds[df_preds['batch_num'].notna()]['A_smiles'].values
    r2_test = df_preds[df_preds['batch_num'].notna()]['B_smiles'].values

    # Get fp list (or lists)
    if concat:
        reacts_fp = utils.fp_list_from_two_smiles_list(r1_test, r2_test)
    else:
        react1_fp = utils.fp_list_from_smiles_list(r1_test)
        react2_fp = utils.fp_list_from_smiles_list(r2_test)
    pca = PCA(n_components=10)
    # Get coordinates and plot
    plt.figure(figsize=(10, 8))
    if concat:
        crds = pca.fit_transform(reacts_fp)
        crds_tsne = TSNE(n_components=2).fit_transform(crds)
        reacts_df = pd.DataFrame(crds_tsne, columns=["X", "Y"])
        reacts_df['batch_num'] = df_preds[df_preds['batch_num'].notna()]['batch_num'].values
        reacts_df['react1_smiles'] = r1_test
        reacts_df['react2_smiles'] = r2_test

        # Compute average X and Y coordinates for each unique react1 and react2 smiles combo.
        averaged_coords = reacts_df.groupby(['react1_smiles', 'react2_smiles']).agg(
            {'X': 'mean', 'Y': 'mean'}).reset_index()
        # Merge the averaged coordinates back to the original dataframe
        plot_df = reacts_df.drop(columns=['X', 'Y']).merge(averaged_coords, on=['react1_smiles', 'react2_smiles'],
                                                           how='left')

        # Plot
        # sns.scatterplot(data=plot_df, x="X", y="Y", hue='batch_num', palette="bright")

        plot_df['batch_num'] = plot_df['batch_num'].astype(str)
        fig = px.scatter(plot_df, x="X", y="Y", color="batch_num", hover_data=['react1_smiles', 'react2_smiles'])
        fig.update_traces(marker={'size': 15})
        fig.write_html("test_scatter.html")

    else:
        # Plot 1. Keep seperate fp of reactants.
        r1crds = pca.fit_transform(react1_fp)
        # r1crds_df = pd.DataFrame(r1crds, columns=["X", "Y"])
        # r1crds_df['batch_num'] = df_preds[df_preds['batch_num'].notna()]['batch_num'].values

        r1crds_embedded = TSNE(n_components=2).fit_transform(r1crds)
        r2crds = pca.fit_transform(react2_fp)
        # r2crds_df = pd.DataFrame(r2crds, columns=["X", "Y"])

        r2crds_embedded = TSNE(n_components=2).fit_transform(r2crds)

        r1tsne_df = pd.DataFrame(r1crds_embedded, columns=["X", "Y"])
        r1tsne_df['batch_num'] = df_preds[df_preds['batch_num'].notna()]['batch_num'].values
        r1tsne_df['smiles'] = df_preds[df_preds['batch_num'].notna()]['A_smiles'].values

        r2tsne_df = pd.DataFrame(r2crds_embedded, columns=["X", "Y"])
        r2tsne_df['batch_num'] = df_preds[df_preds['batch_num'].notna()]['batch_num'].values

        print(r1tsne_df)

        # Compute average X and Y coordinates for each unique 'smiles'
        averaged_coords = r1tsne_df.groupby('smiles').agg({'X': 'mean', 'Y': 'mean'}).reset_index()

        # Merge the averaged coordinates back to the original dataframe
        r1tsne_df_updated = r1tsne_df.drop(columns=['X', 'Y']).merge(averaged_coords, on='smiles', how='left')
        print(r1tsne_df_updated)

        # ax = sns.scatterplot(data=r1crds_df, x="X", y="Y", hue='batch_num', palette="husl")
        ax1 = sns.scatterplot(data=r1tsne_df_updated, x="X", y="Y", hue='batch_num', palette="bright")
        ax2 = sns.scatterplot(data=r2tsne_df, x="X", y="Y", hue='batch_num', palette='bright', marker='s', legend=False)

        # Create custom legend handles
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Amines', markersize=10, markerfacecolor='black'),
            Line2D([0], [0], marker='s', color='w', label='Aryl Halides', markersize=10, markerfacecolor='black')
        ]

        # Add the custom legend to the current legend
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles + legend_elements, labels=labels + ['Amines', 'Aryl Halides'], title='Batch Number')

    plt.show()
    fig.show()
    return plot_df