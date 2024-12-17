import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Tuple, Dict
import os

class RecommenderSystem:
    def __init__(self, dataDirectory: str):
        """Initialize the recommender system with the path to the dataset directory."""
        self.dataDirectory = dataDirectory
        self.userItemMtrix = None
        self.trainData = None
        self.testData = None
        self.moviesData = None
        
    def loadMovielensData(self) -> pd.DataFrame:
        """Load and preprocess MovieLens dataset."""
        # Load ratings and movies data
        ratingsPath = os.path.join(self.dataDirectory, 'ratings.csv')
        moviesPath = os.path.join(self.dataDirectory, 'movies.csv')
       
        ratingsDf = pd.read_csv(ratingsPath)
        self.moviesData = pd.read_csv(moviesPath)
        
        #Takes needed columns from ratings files
        ratingsDf = ratingsDf[['userId', 'movieId', 'rating']]
        return ratingsDf
    
    def preprocessData(self) -> None:
        """Preprocess the data and split into training and testing sets."""
        
        df = self.loadMovielensData()
        
        # Split data for each user
        trainData = []
        testData = []
        
        for user in df['userId'].unique():
            userData = df[df['userId'] == user]
            userTrain, userTest = train_test_split(userData, test_size=0.2, random_state=42)
            trainData.append(userTrain)
            testData.append(userTest)
            
        self.trainData = pd.concat(trainData)
        self.testData = pd.concat(testData)
        
        # Create user-item matrix
        self.userItemMatrix = pd.pivot_table(
            self.trainData,
            values='rating',
            index='userId',
            columns='movieId',
            fill_value=0
        ).values
        
    def matrixFactorization(self, k: int = 50) -> None:
        """Perform matrix factorization using SVD."""
        # Normalize the ratings
        userRatingsMean = np.mean(self.userItemMatrix, axis=1)
        normalizedMatrix = self.userItemMatrix - userRatingsMean.reshape(-1, 1)
        
        # Perform SVD
        U, sigma, Vt = svds(normalizedMatrix, k=k)
        sigma = np.diag(sigma)
        
        # Calculate predicted ratings
        self.predictedRatings = np.dot(np.dot(U, sigma), Vt) + userRatingsMean.reshape(-1, 1)
    
    def predictRating(self, userId: int, movieId: int) -> float:
        """Predict rating for a specific user-movie pair."""
        userIdx = np.where(self.trainData['userId'].unique() == userId)[0][0]
        movieIdx = np.where(self.trainData['movieId'].unique() == movieId)[0][0]
        return self.predictedRatings[userIdx, movieIdx]
    
    def calculateMetrics(self) -> Tuple[float, float]:
        """Calculate MAE and RMSE for the test set."""
        testPredictions = []
        testActual = []
        
        uniqueUsers = self.trainData['userId'].unique()
        uniqueMovies = self.trainData['movieId'].unique()
        
        for _, row in self.testData.iterrows():
            try:
                userIdx = np.where(uniqueUsers == row['userId'])[0][0]
                movieIdx = np.where(uniqueMovies == row['movieId'])[0][0]
                predicted = self.predictedRatings[userIdx, movieIdx]
                testPredictions.append(predicted)
                testActual.append(row['rating'])
            except IndexError:
                continue
            
        mae = mean_absolute_error(testActual, testPredictions)
        rmse = np.sqrt(mean_squared_error(testActual, testPredictions))
        
        return mae, rmse
    
    def generateRecommendations(self, userId: int, n: int = 10) -> List[Tuple[int, str, float]]:
        """Generate top-N recommendations for a user."""
        try:
            userIdx = np.where(self.trainData['userId'].unique() == userId)[0][0]
            
            # Get user's rated movies
            userRatedMovies = set(self.trainData[self.trainData['userId'] == userId]['movieId'])
            
            # Get predictions for all movies
            userPredictions = self.predictedRatings[userIdx]
            movieIds = self.trainData['movieId'].unique()
            
            # Create movie-prediction pairs for unrated movies
            candidates = [(movieId, pred) for movieId, pred in zip(movieIds, userPredictions) 
                         if movieId not in userRatedMovies]
            
            # Sort by predicted rating and get top N
            recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
            
            # Add movie titles to recommendations
            recommendationsWithTitles = []
            for movieId, predRating in recommendations:
                movieTitle = self.moviesData[self.moviesData['movieId'] == movieId]['title'].iloc[0]
                recommendationsWithTitles.append((movieId, movieTitle, predRating))
            
            return recommendationsWithTitles
        except IndexError:
            print(f"Error: User {userId} not found in training data.")
            return []
    
    def calculateRecommendationMetrics(self) -> Dict[str, float]:
        """Calculate Precision, Recall, F-measure, and NDCG for recommendations."""
        precisions = []
        recalls = []
        ndcgs = []
        
        for userId in self.testData['userId'].unique():
            # Get actual movies in test set
            actualMovies = set(self.testData[self.testData['userId'] == userId]['movieId'])
            
            # Get recommended movies
            recommendations = self.generateRecommendations(userId)
            if not recommendations:
                continue
                
            recommendedMovies = set(movieId for movieId, _, _ in recommendations)
            
            # Calculate precision and recall
            hits = len(actualMovies.intersection(recommendedMovies))
            precision = hits / 10  # 10 is recommendation list length
            recall = hits / len(actualMovies) if actualMovies else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate NDCG
            dcg = 0
            idcg = 0
            for i, (movieId, _, _) in enumerate(recommendations):
                if movieId in actualMovies:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(len(actualMovies), 10)):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        # Calculate averages
        avgPrecision = np.mean(precisions)
        avgRecall = np.mean(recalls)
        fMeasure = 2 * avgPrecision * avgRecall / (avgPrecision + avgRecall) if (avgPrecision + avgRecall) > 0 else 0
        avgNdcg = np.mean(ndcgs)
        
        return {
            'precision': avgPrecision,
            'recall': avgRecall,
            'fMeasure': fMeasure,
            'ndcg': avgNdcg
        }
    
    def printRecommendationsTable(self, userId: int, n: int = 10) -> None:
        """Print recommendations in a formatted table."""
        recommendations = self.generateRecommendations(userId, n)
        if not recommendations:
            return
            
        # Print table header
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*18 + "+" + "-"*6 + "+")
        print("|user_id|movie_Id    |predicted_rate|rank|")
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*18 + "+" + "-"*6 + "+")
        
        # Print recommendations
        for rank, (movieId, _, predRating) in enumerate(recommendations, 1):
            print(f"|{userId:>8}|{movieId:>9}|{predRating:>16.1f}|{rank:>4}|")
        
        # Print table footer
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*18 + "+" + "-"*6 + "+")
    


def main():
    # Use relative path to data directory
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(os.path.dirname(currentDirectory), 'data')
    
    recommender = RecommenderSystem(dataDirectory)
    
    # Preprocess data
    recommender.preprocessData()
    
    # Train model
    recommender.matrixFactorization(k=50)

    # Just print the table
    userId = recommender.trainData['userId'].iloc[0]
    recommender.printRecommendationsTable(userId)
    
    # Explain recommendations for a specific user
    print(f"\nExplanation for Recommendations for User {userId}:")
    recommender.explainRecommendations(userId)

    # Calculate rating prediction metrics
    mae, rmse = recommender.calculateMetrics()
    print(f"\nRating Prediction Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Calculate recommendation metrics
    metrics = recommender.calculateRecommendationMetrics()
    print(f"\nRecommendation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F-measure: {metrics['fMeasure']:.4f}")
    print(f"NDCG: {metrics['ndcg']:.4f}")
    
    # Example: Generate recommendations for a specific user
    userId = recommender.trainData['userId'].iloc[0]  # Get first user as example
    print(f"\nTop 10 Recommendations for User {userId}:")
    recommendations = recommender.generateRecommendations(userId)
    for movieId, title, predRating in recommendations:
        print(f"{title}: {predRating:.2f}")
    
def main():
    # Use relative path to data directory
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(os.path.dirname(currentDirectory), 'data')
    
    recommender = RecommenderSystem(dataDirectory)
    
    # Preprocess data
    recommender.preprocessData()
    
    # Train model
    recommender.matrixFactorization(k=50)

    #Print the table
    userId = recommender.trainData['userId'].iloc[0]
    recommender.printRecommendationsTable(userId)

    # Calculate rating prediction metrics
    mae, rmse = recommender.calculateMetrics()
    print(f"\nRating Prediction Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Calculate recommendation metrics
    metrics = recommender.calculateRecommendationMetrics()
    print(f"\nRecommendation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F-measure: {metrics['fMeasure']:.4f}")
    print(f"NDCG: {metrics['ndcg']:.4f}")
    
    # Example: Generate recommendations for a specific user
    userId = recommender.trainData['userId'].iloc[0]  # Get first user as example
    print(f"\nTop 10 Recommendations for User {userId}:")
    recommendations = recommender.generateRecommendations(userId)
    for movieId, title, predRating in recommendations:
        print(f"{title}: {predRating:.2f}")

if __name__ == "__main__":
    main()

