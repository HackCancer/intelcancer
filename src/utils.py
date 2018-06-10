def clean_mutations(mutations):
    mutations.drop('Gene name', axis=1, inplace=True)
    mutations.drop('Accession Number', axis=1, inplace=True)
    mutations.drop('Gene CDS length', axis=1, inplace=True)
    mutations.drop('HGNC ID', axis=1, inplace=True)
    mutations.drop('Sample name', axis=1, inplace=True)
    mutations.drop('ID_tumour', axis=1, inplace=True)
    mutations.drop('Primary site', axis=1, inplace=True)
    mutations.drop('Site subtype 1', axis=1, inplace=True)
    mutations.drop('Site subtype 2', axis=1, inplace=True)
    mutations.drop('Site subtype 3', axis=1, inplace=True)
    mutations.drop('Primary histology', axis=1, inplace=True)
    mutations.drop('Histology subtype 1', axis=1, inplace=True)
    mutations.drop('Histology subtype 2', axis=1, inplace=True)
    mutations.drop('Histology subtype 3', axis=1, inplace=True)
    mutations.drop('Genome-wide screen', axis=1, inplace=True)
    mutations.drop('Mutation CDS', axis=1, inplace=True)
    mutations.drop('Mutation AA', axis=1, inplace=True)
    mutations.drop('Mutation Description', axis=1, inplace=True)
    mutations.drop('Mutation zygosity', axis=1, inplace=True)
    mutations.drop('LOH', axis=1, inplace=True)
    mutations.drop('GRCh', axis=1, inplace=True)
    mutations.drop('Mutation genome position', axis=1, inplace=True)
    mutations.drop('Mutation strand', axis=1, inplace=True)
    mutations.drop('SNP', axis=1, inplace=True)
    # mutations.drop('Resistance Mutation', axis=1, inplace=True)
    mutations.drop('FATHMM score', axis=1, inplace=True)
    mutations.drop('Mutation somatic status', axis=1, inplace=True)
    mutations.drop('Pubmed_PMID', axis=1, inplace=True)
    mutations.drop('ID_STUDY', axis=1, inplace=True)
    mutations.drop('Sample Type', axis=1, inplace=True)
    mutations.drop('Tumour origin', axis=1, inplace=True)
    mutations.drop('Tier', axis=1, inplace=True)