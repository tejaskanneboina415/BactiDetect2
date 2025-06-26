# Microbio AI - Dataset Expansion Plan

## Current State
- 5 species: B.subtilis, C.albicans, E.coli, P.aeruginosa, S.aureus
- ~10 images per species
- Basic ResNet50 model

## Target State
- 500+ microbial species
- 10,000+ images
- Advanced model architecture
- Hierarchical classification

## Phase 1: Data Collection Strategy

### 1.1 Public Datasets to Integrate
- **Bacterial Visions**: 1,000+ bacterial species
- **FungiDB**: 2,000+ fungal species
- **MicrobeNet**: Clinical microbiology database
- **NCBI Taxonomy**: Reference database
- **iNaturalist**: Community-contributed images
- **PubMed Central**: Research paper images

### 1.2 Data Sources
- Scientific publications
- Medical databases
- Research institutions
- Crowdsourced platforms
- Museum collections

### 1.3 Target Species Categories
- **Bacteria**: 300+ species (Gram-positive, Gram-negative, acid-fast)
- **Fungi**: 150+ species (yeasts, molds, dimorphic)
- **Protozoa**: 50+ species
- **Archaea**: 20+ species
- **Algae**: 30+ species

## Phase 2: Model Architecture Upgrades

### 2.1 Advanced Models
- Vision Transformer (ViT)
- EfficientNetV2
- ConvNeXt
- Swin Transformer

### 2.2 Hierarchical Classification
- Domain level (Bacteria, Archaea, Eukarya)
- Phylum level
- Class level
- Order level
- Family level
- Genus level
- Species level

### 2.3 Multi-task Learning
- Species identification
- Gram staining prediction
- Colony morphology
- Growth characteristics

## Phase 3: Implementation Steps

1. **Data Collection Scripts**
2. **Data Preprocessing Pipeline**
3. **Model Training Infrastructure**
4. **Evaluation Framework**
5. **Web Interface Updates**

## Phase 4: Quality Assurance

- Expert validation
- Cross-referencing with databases
- Image quality assessment
- Taxonomic accuracy verification 