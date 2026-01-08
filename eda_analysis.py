# """
# Butterfly Classification - Exploratory Data Analysis
# Analyze dataset distribution, visualize samples, and identify issues
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# import warnings
# warnings.filterwarnings('ignore')

# # Set style
# sns.set_style('whitegrid')
# plt.rcParams['figure.figsize'] = (12, 8)

# class ButterflyEDA:
#     """
#     Exploratory Data Analysis for Butterfly Dataset
#     """
    
#     def __init__(self, csv_path):
#         self.df = pd.read_csv(csv_path)
#         print(f"Dataset loaded: {self.df.shape[0]} images, {self.df.shape[1]} columns")
        
#     def analyze_class_distribution(self):
#         """
#         Analyze and visualize class distribution
#         """
#         class_counts = self.df['label'].value_counts().sort_values(ascending=False)
        
#         print("\n" + "=" * 60)
#         print("CLASS DISTRIBUTION ANALYSIS")
#         print("=" * 60)
#         print(f"Total number of classes: {self.df['label'].nunique()}")
#         print(f"Total images: {len(self.df)}")
#         print(f"\nImages per class:")
#         print(f"  Min: {class_counts.min()}")
#         print(f"  Max: {class_counts.max()}")
#         print(f"  Mean: {class_counts.mean():.2f}")
#         print(f"  Median: {class_counts.median()}")
#         print(f"  Std Dev: {class_counts.std():.2f}")
        
#         # Check for imbalance
#         imbalance_ratio = class_counts.max() / class_counts.min()
#         print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
#         if imbalance_ratio > 3:
#             print("⚠️  WARNING: Significant class imbalance detected!")
#             print("   Consider using class weights or data augmentation")
        
#         # Visualization
#         fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
#         # Bar plot
#         class_counts.plot(kind='bar', ax=axes[0], color='steelblue')
#         axes[0].set_title('Distribution of Butterfly Species (All Classes)', fontsize=14, fontweight='bold')
#         axes[0].set_xlabel('Species')
#         axes[0].set_ylabel('Number of Images')
#         axes[0].axhline(y=class_counts.mean(), color='red', linestyle='--', label=f'Mean: {class_counts.mean():.0f}')
#         axes[0].legend()
#         axes[0].tick_params(axis='x', rotation=90, labelsize=6)
        
#         # Distribution histogram
#         axes[1].hist(class_counts, bins=20, color='coral', edgecolor='black')
#         axes[1].set_title('Histogram of Images per Class', fontsize=14, fontweight='bold')
#         axes[1].set_xlabel('Number of Images per Class')
#         axes[1].set_ylabel('Frequency (Number of Classes)')
#         axes[1].axvline(x=class_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {class_counts.mean():.0f}')
#         axes[1].legend()
        
#         plt.tight_layout()
#         plt.savefig('reports/class_distribution.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # Top and bottom classes
#         print("\nTop 10 classes with most images:")
#         print(class_counts.head(10))
        
#         print("\nBottom 10 classes with fewest images:")
#         print(class_counts.tail(10))
        
#         return class_counts
    
#     def visualize_sample_images(self, samples_per_class=3):
#         """
#         Display sample images from different classes
#         """
#         print("\n" + "=" * 60)
#         print("SAMPLE IMAGE VISUALIZATION")
#         print("=" * 60)
        
#         # Select random classes
#         num_classes_to_show = 12
#         random_classes = np.random.choice(self.df['label'].unique(), num_classes_to_show, replace=False)
        
#         fig, axes = plt.subplots(num_classes_to_show, samples_per_class, figsize=(15, 25))
        
#         for idx, class_name in enumerate(random_classes):
#             class_images = self.df[self.df['label'] == class_name].sample(n=min(samples_per_class, len(self.df[self.df['label'] == class_name])))
            
#             for img_idx, (_, row) in enumerate(class_images.iterrows()):
#                 try:
#                     img = Image.open(row['filename'])
#                     axes[idx, img_idx].imshow(img)
#                     axes[idx, img_idx].axis('off')
                    
#                     if img_idx == 0:
#                         axes[idx, img_idx].set_title(f"{class_name}", fontsize=10, fontweight='bold')
#                 except Exception as e:
#                     axes[idx, img_idx].text(0.5, 0.5, 'Error loading image', 
#                                            ha='center', va='center')
#                     axes[idx, img_idx].axis('off')
        
#         plt.suptitle('Sample Images from Random Butterfly Species', fontsize=16, fontweight='bold', y=0.995)
#         plt.tight_layout()
#         plt.savefig('reports/sample_images_grid.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Sample grid saved with {num_classes_to_show} classes × {samples_per_class} images")
    
#     def check_image_properties(self, sample_size=100):
#         """
#         Analyze image properties (dimensions, formats, etc.)
#         """
#         print("\n" + "=" * 60)
#         print("IMAGE PROPERTIES ANALYSIS")
#         print("=" * 60)
        
#         sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        
#         widths = []
#         heights = []
#         aspect_ratios = []
#         formats = []
#         corrupted = []
        
#         for _, row in sample_df.iterrows():
#             try:
#                 img = Image.open(row['filename'])
#                 widths.append(img.width)
#                 heights.append(img.height)
#                 aspect_ratios.append(img.width / img.height)
#                 formats.append(img.format)
#             except Exception as e:
#                 corrupted.append(row['filename'])
        
#         print(f"\nAnalyzed {len(widths)} images")
#         print(f"\nImage Dimensions:")
#         print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
#         print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")
#         print(f"\nAspect Ratios:")
#         print(f"  Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}")
#         print(f"\nImage Formats:")
#         print(pd.Series(formats).value_counts())
        
#         if corrupted:
#             print(f"\n⚠️  WARNING: {len(corrupted)} corrupted images found!")
#             print("Corrupted images:", corrupted[:5])
#         else:
#             print("\n✓ No corrupted images detected in sample")
        
#         # Visualize distributions
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
#         axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black')
#         axes[0].set_title('Image Width Distribution')
#         axes[0].set_xlabel('Width (pixels)')
#         axes[0].set_ylabel('Frequency')
        
#         axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
#         axes[1].set_title('Image Height Distribution')
#         axes[1].set_xlabel('Height (pixels)')
#         axes[1].set_ylabel('Frequency')
        
#         axes[2].hist(aspect_ratios, bins=30, color='lightgreen', edgecolor='black')
#         axes[2].set_title('Aspect Ratio Distribution')
#         axes[2].set_xlabel('Aspect Ratio (W/H)')
#         axes[2].set_ylabel('Frequency')
        
#         plt.tight_layout()
#         plt.savefig('reports/image_properties.png', dpi=300, bbox_inches='tight')
#         plt.close()
    
#     def generate_summary_report(self):
#         """
#         Generate comprehensive summary report
#         """
#         print("\n" + "=" * 60)
#         print("DATASET SUMMARY REPORT")
#         print("=" * 60)
        
#         summary = {
#             'Total Images': len(self.df),
#             'Number of Classes': self.df['label'].nunique(),
#             'Columns': list(self.df.columns),
#             'Missing Values': self.df.isnull().sum().to_dict(),
#             'Data Types': self.df.dtypes.to_dict()
#         }
        
#         print("\nDataset Overview:")
#         for key, value in summary.items():
#             print(f"  {key}: {value}")
        
#         print("\nFirst few rows:")
#         print(self.df.head())
        
#         print("\nDataset Info:")
#         self.df.info()
        
#         return summary


# def main():
#     """
#     Main execution function for EDA
#     """
#     print("=" * 60)
#     print("Butterfly Species Classification - EDA")
#     print("=" * 60)
    
#     # Create reports directory
#     os.makedirs('reports', exist_ok=True)
    
#     # Initialize EDA
#     eda = ButterflyEDA('data/Training_set.csv')
    
#     # Run analyses
#     print("\n[1] Analyzing class distribution...")
#     class_counts = eda.analyze_class_distribution()
    
#     print("\n[2] Visualizing sample images...")
#     eda.visualize_sample_images(samples_per_class=3)
    
#     print("\n[3] Checking image properties...")
#     eda.check_image_properties(sample_size=100)
    
#     print("\n[4] Generating summary report...")
#     summary = eda.generate_summary_report()
    
#     print("\n" + "=" * 60)
#     print("EDA Complete! Reports saved in 'reports/' directory")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()






"""
Butterfly Classification - Exploratory Data Analysis
Analyze dataset distribution, visualize samples, and identify issues
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class ButterflyEDA:
    """
    Exploratory Data Analysis for Butterfly Dataset
    """

    def __init__(self, csv_path, image_base_dir='train'):
        self.df = pd.read_csv(csv_path)
        self.image_base_dir = image_base_dir

        # Correct path construction (NO subfolders)
        self.df['filepath'] = self.df['filename'].apply(
            lambda x: os.path.join(image_base_dir, x)
        )

        print(f"Dataset loaded: {len(self.df)} images")
        
    def analyze_class_distribution(self):
        """
        Analyze and visualize class distribution
        """
        class_counts = self.df['label'].value_counts().sort_values(ascending=False)
        
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"Total number of classes: {self.df['label'].nunique()}")
        print(f"Total images: {len(self.df)}")
        print(f"\nImages per class:")
        print(f"  Min: {class_counts.min()}")
        print(f"  Max: {class_counts.max()}")
        print(f"  Mean: {class_counts.mean():.2f}")
        print(f"  Median: {class_counts.median()}")
        print(f"  Std Dev: {class_counts.std():.2f}")
        
        # Check for imbalance
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("⚠️  WARNING: Significant class imbalance detected!")
            print("   Consider using class weights or data augmentation")
        else:
            print("✓ Class distribution is relatively balanced")
        
        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Distribution of Butterfly Species (All Classes)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Species')
        axes[0].set_ylabel('Number of Images')
        axes[0].axhline(y=class_counts.mean(), color='red', linestyle='--', label=f'Mean: {class_counts.mean():.0f}')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=90, labelsize=6)
        
        # Distribution histogram
        axes[1].hist(class_counts, bins=20, color='coral', edgecolor='black')
        axes[1].set_title('Histogram of Images per Class', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Images per Class')
        axes[1].set_ylabel('Frequency (Number of Classes)')
        axes[1].axvline(x=class_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {class_counts.mean():.0f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('reports/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: reports/class_distribution.png")
        
        # Top and bottom classes
        print("\nTop 10 classes with most images:")
        print(class_counts.head(10))
        
        print("\nBottom 10 classes with fewest images:")
        print(class_counts.tail(10))
        
        return class_counts
    
    def visualize_sample_images(self, samples_per_class=3):
        """
        Display sample images from different classes
        """
        print("\n" + "=" * 60)
        print("SAMPLE IMAGE VISUALIZATION")
        print("=" * 60)
        
        # Select random classes
        num_classes_to_show = 12
        random_classes = np.random.choice(self.df['label'].unique(), num_classes_to_show, replace=False)
        
        fig, axes = plt.subplots(num_classes_to_show, samples_per_class, figsize=(15, 25))
        
        loaded_count = 0
        error_count = 0
        
        for idx, class_name in enumerate(random_classes):
            class_images = self.df[self.df['label'] == class_name].sample(
                n=min(samples_per_class, len(self.df[self.df['label'] == class_name]))
            )
            
            for img_idx, (_, row) in enumerate(class_images.iterrows()):
                try:
                    img_path = row['filepath']
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        axes[idx, img_idx].imshow(img)
                        axes[idx, img_idx].axis('off')
                        loaded_count += 1
                        
                        if img_idx == 0:
                            # Truncate long names
                            display_name = class_name[:30] + '...' if len(class_name) > 30 else class_name
                            axes[idx, img_idx].set_title(display_name, fontsize=9, fontweight='bold')
                    else:
                        axes[idx, img_idx].text(0.5, 0.5, 'File not found', 
                                               ha='center', va='center', fontsize=8)
                        axes[idx, img_idx].axis('off')
                        error_count += 1
                        
                except Exception as e:
                    axes[idx, img_idx].text(0.5, 0.5, 'Error loading', 
                                           ha='center', va='center', fontsize=8)
                    axes[idx, img_idx].axis('off')
                    error_count += 1
        
        plt.suptitle('Sample Images from Random Butterfly Species', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('reports/sample_images_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Sample grid saved: {num_classes_to_show} classes × {samples_per_class} images")
        print(f"  Successfully loaded: {loaded_count} images")
        if error_count > 0:
            print(f"  ⚠️  Errors: {error_count} images")
    
    def check_image_properties(self, sample_size=100):
        """
        Analyze image properties (dimensions, formats, etc.)
        """
        print("\n" + "=" * 60)
        print("IMAGE PROPERTIES ANALYSIS")
        print("=" * 60)
        
        sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        
        widths = []
        heights = []
        aspect_ratios = []
        formats = []
        file_sizes = []
        corrupted = []
        missing = []
        
        for _, row in sample_df.iterrows():
            try:
                img_path = row['filepath']
                
                # Check if file exists
                if not os.path.exists(img_path):
                    missing.append(img_path)
                    continue
                
                # Try to open and analyze image
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
                aspect_ratios.append(img.width / img.height)
                formats.append(img.format if img.format else 'Unknown')
                
                # Get file size in KB
                file_sizes.append(os.path.getsize(img_path) / 1024)
                
            except Exception as e:
                corrupted.append(img_path)
        
        # Check if we have data to analyze
        if len(widths) == 0:
            print(f"\n⚠️  ERROR: Could not analyze any images!")
            print(f"  Missing files: {len(missing)}")
            print(f"  Corrupted files: {len(corrupted)}")
            
            if missing:
                print("\nSample missing files:")
                for f in missing[:3]:
                    print(f"  {f}")
                    
            print("\n💡 Make sure:")
            print(f"  1. Images are in: {self.image_base_dir}/SPECIES_NAME/filename.jpg")
            print(f"  2. CSV 'label' matches folder names exactly")
            print(f"  3. CSV 'filename' matches actual filenames")
            return
        
        print(f"\nAnalyzed {len(widths)} images (out of {len(sample_df)} sampled)")
        
        print(f"\nImage Dimensions:")
        print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
        print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")
        
        print(f"\nAspect Ratios:")
        print(f"  Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}")
        
        print(f"\nFile Sizes (KB):")
        print(f"  Min: {min(file_sizes):.1f}, Max: {max(file_sizes):.1f}, Mean: {np.mean(file_sizes):.1f}")
        
        print(f"\nImage Formats:")
        format_counts = pd.Series(formats).value_counts()
        for fmt, count in format_counts.items():
            print(f"  {fmt}: {count}")
        
        if missing:
            print(f"\n⚠️  WARNING: {len(missing)} missing files in sample!")
        
        if corrupted:
            print(f"\n⚠️  WARNING: {len(corrupted)} corrupted images in sample!")
            
        if not missing and not corrupted:
            print("\n✓ No corrupted or missing images detected in sample")
        
        # Visualize distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(widths, bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(aspect_ratios, bins=30, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(file_sizes, bins=30, color='plum', edgecolor='black')
        axes[1, 1].set_title('File Size Distribution')
        axes[1, 1].set_xlabel('File Size (KB)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('reports/image_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: reports/image_properties.png")
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("\n" + "=" * 60)
        print("DATASET SUMMARY REPORT")
        print("=" * 60)
        
        summary = {
            'Total Images': len(self.df),
            'Number of Classes': self.df['label'].nunique(),
            'Columns': list(self.df.columns),
            'Missing Values': self.df.isnull().sum().to_dict(),
            'Data Types': self.df.dtypes.to_dict()
        }
        
        print("\nDataset Overview:")
        for key, value in summary.items():
            if key not in ['Data Types', 'Missing Values']:
                print(f"  {key}: {value}")
        
        print("\nFirst few rows:")
        print(self.df[['filename', 'label']].head(10))
        
        # Check if image files exist
        existing_files = sum(1 for path in self.df['filepath'] if os.path.exists(path))
        print(f"\nFile Existence Check:")
        print(f"  Files found: {existing_files}/{len(self.df)} ({existing_files/len(self.df)*100:.1f}%)")
        
        return summary


def main():
    """
    Main execution function for EDA
    """
    print("=" * 60)
    print("Butterfly Species Classification - EDA")
    print("=" * 60)
    
    # Configuration - ADJUST THESE AS NEEDED
    CSV_PATH = 'data/Training_set.csv'  # or 'data/Training_set.csv'
    IMAGE_BASE_DIR = 'data/train'        # Base directory with species folders
    
    print(f"\nConfiguration:")
    print(f"  CSV: {CSV_PATH}")
    print(f"  Images: {IMAGE_BASE_DIR}/")
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Initialize EDA
    eda = ButterflyEDA(CSV_PATH, IMAGE_BASE_DIR)
    
    # Run analyses
    print("\n[1] Analyzing class distribution...")
    class_counts = eda.analyze_class_distribution()
    
    print("\n[2] Visualizing sample images...")
    eda.visualize_sample_images(samples_per_class=3)
    
    print("\n[3] Checking image properties...")
    eda.check_image_properties(sample_size=100)
    
    print("\n[4] Generating summary report...")
    summary = eda.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("EDA Complete! Reports saved in 'reports/' directory")
    print("=" * 60)
    print("\nGenerated files:")
    print("  ✓ reports/class_distribution.png")
    print("  ✓ reports/sample_images_grid.png")
    print("  ✓ reports/image_properties.png")


if __name__ == "__main__":
    main()