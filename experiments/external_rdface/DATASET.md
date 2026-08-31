# RDFace: what the source paper says

Feng G, Long Y, Ali H, Lou E, Butt F, Liu Q, Wang Y, Hu P.
*RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under
Extreme Data Scarcity and Phenotype-Aware Synthetic Generation.*
arXiv:2604.03454. Project page: https://github.com/Kkathyf/RDFace

Recorded here so the provenance facts do not have to be re-read out of the PDF
(`raw_data/RDFace.pdf`, 32 pages including supplementary).

## Scale and structure

456 pediatric facial images over 103 rare genetic conditions, on average 4.4
per condition (1 to 7 in practice). Released as `rd_images/<ABBR>/<ABBR>.<n>.png`
plus a root `disease_images.csv`. The archive also carries `__MACOSX/`
resource forks and a `.DS_Store`, which have to be excluded on extraction.

## What the metadata contains, and what it does not

The released CSV has exactly six fields: image name, disease name, gene,
disease abbreviation, disease subcategory, Orphanet code.

**There are no HPO annotations.** The string "HPO" does not occur anywhere in
the paper. `sub_category` looks phenotype-like but is a disease-level
classification, one value per disease rather than per image, and none of its 78
values names a facial feature: they are systemic (abdominal distention,
arrhythmia, ataxia, cardiomyopathy, intellectual disability, muscular
dystrophy). No term-level validation is possible on this dataset.

**There are no per-image demographics.** Age, sex and country exist at the
dataset level only (Figure 2), and the paper states they are "summarized at the
dataset level but not shared as metadata due to inconsistent availability".

**There is no ancestry.** The authors say so directly: population-level
demographic attributes such as ethnicity, ancestry or skin tone "are not
available in our dataset, as web-scraped rare disease case reports rarely
provide standardized annotations". They fall back on geographic region as a
proxy for their own bias analysis, and that region label is not released
either.

## Age

Dataset-level average **6.36 years**, range 0 to 18, curated with emphasis on
children under 12 "to minimize visual confounding from adult comorbidities".

For comparison, the GMDB frontal patients with a usable age have a median of
6.0 years. The two cohorts are therefore close in overall age, which bounds how
much a difference in age composition can explain in a cross-cohort comparison.
Per-disease age remains uncontrolled, since no image carries an age.

## Selection: the images were already screened as frontal

Inclusion required images to be "frontal-facing, portrait-style, and of
sufficient resolution, showing a single patient with open eyes", sourced by
structured web search over peer-reviewed literature, hospital foundations and
verified clinical reports, with manually reviewed advocacy content where
necessary. Curation was supervised by clinical geneticists and two clinical
fellows independently reviewed the image-label associations.

This changes how the pose-gate rate should be read. FaceKit's geometric
criterion rejects 23% of these images, so the number is not "77% of the images
are usable" but "the geometric criterion disagrees with a human frontality
judgement on nearly a quarter of images".

## Patient identity: not established, by them or by us

The paper never states that the images within one disease class are distinct
individuals, gives no patient count, and describes no deduplication step. The
indirect evidence is that each image must show "a single patient", that
Figure 2 counts "cases" across 46 countries, and that images were gathered from
independent publications and sites.

So the possibility that one folder holds several photographs of one patient is
open in the source dataset itself. A difference hash finds no repeated image
here, but has no discriminative power on this corpus at all, so it settles only
that no image appears twice.

## Resolution

Short sides run from 41 to 837 px, median 197. The authors resize to 224 x 224
for their own benchmarks, the same working resolution as the GMDB images.

## Ethics

Approved by the Western University Health Science Research Ethics Board
(reference 2023-122744-77394); images come from publicly available sources.
