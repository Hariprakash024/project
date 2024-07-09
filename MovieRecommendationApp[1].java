import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.*;

class Node {
    String name;
    String genre; // Only for movie nodes
    Set<Node> connections;

    public Node(String name, String genre) {
        this.name = name;
        this.genre = genre;
        this.connections = new HashSet<>();
    }

    public void addConnection(Node node) {
        connections.add(node);
    }

    public Set<Node> getConnections() {
        return connections;
    }

    @Override
    public String toString() {
        return name;
    }
}

class Graph {
    Map<String, Node> nodes;

    public Graph() {
        nodes = new HashMap<>();
    }

    public void addNode(String name, String genre) {
        nodes.put(name, new Node(name, genre));
    }

    public void addEdge(String userName, String movieName) {
        Node user = nodes.get(userName);
        Node movie = nodes.get(movieName);
        if (user != null && movie != null) {
            user.addConnection(movie);
            movie.addConnection(user);
        }
    }

    public List<String> recommendMovies(String userName, List<String> defaultMovies) {
        Node user = nodes.get(userName);
        if (user == null) {
            return defaultMovies;
        }

        Map<String, Integer> genreCount = new HashMap<>();
        Set<Node> watchedMovies = user.getConnections();

        for (Node movie : watchedMovies) {
            genreCount.put(movie.genre, genreCount.getOrDefault(movie.genre, 0) + 1);
        }

        PriorityQueue<Node> recommendations = new PriorityQueue<>((a, b) -> {
            int genreA = genreCount.getOrDefault(a.genre, 0);
            int genreB = genreCount.getOrDefault(b.genre, 0);
            if (genreA == genreB) {
                return new Random().nextInt(2) - 1; // randomize tie-breaker
            }
            return genreB - genreA;
        });

        for (Node movie : nodes.values()) {
            if (!watchedMovies.contains(movie) && movie.genre != null) {
                recommendations.add(movie);
            }
        }

        List<String> result = new ArrayList<>();
        while (!recommendations.isEmpty() && result.size() < 10) { // limit to 10 recommendations
            result.add(recommendations.poll().name);
        }

        // Add some random movies from different genres
        List<String> additionalMovies = new ArrayList<>();
        for (Node movie : nodes.values()) {
            if (!watchedMovies.contains(movie) && movie.genre != null && !result.contains(movie.name)) {
                additionalMovies.add(movie.name);
            }
        }
        Collections.shuffle(additionalMovies);
        for (int i = 0; i < Math.min(5, additionalMovies.size()); i++) { // add up to 5 random movies
            result.add(additionalMovies.get(i));
        }

        if (result.isEmpty()) {
            return defaultMovies;
        }

        return result;
    }
}

public class MovieRecommendationApp extends JFrame {
    private Graph graph;
    private JTextField userInput;
    private JPanel loginPanel;
    private JPanel moviePanel;
    private String loggedInUser;
    private DefaultListModel<String> unwatchedListModel;
    private DefaultListModel<String> watchedListModel;

    public MovieRecommendationApp() {
        graph = new Graph();
        initializeGraph();

        setTitle("Movie Recommendation System");
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Login Panel
        loginPanel = new JPanel();
        userInput = new JTextField(20);
        userInput.setToolTipText("Enter User Name");
        JButton loginButton = new JButton("Login");

        loginButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loggedInUser = userInput.getText();
                if (loggedInUser.isEmpty()) {
                    JOptionPane.showMessageDialog(MovieRecommendationApp.this, "Please enter a user name.");
                    return;
                }
                if (!graph.nodes.containsKey(loggedInUser)) {
                    graph.addNode(loggedInUser, null);
                }
                switchToMoviePanel();
            }
        });

        loginPanel.add(new JLabel("User Name:"));
        loginPanel.add(userInput);
        loginPanel.add(loginButton);

        // Movie Panel
        moviePanel = new JPanel(new GridLayout(1, 2));
        unwatchedListModel = new DefaultListModel<>();
        watchedListModel = new DefaultListModel<>();
        JList<String> unwatchedList = new JList<>(unwatchedListModel);
        JList<String> watchedList = new JList<>(watchedListModel);
        unwatchedList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        unwatchedList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                String selectedMovie = unwatchedList.getSelectedValue();
                if (selectedMovie != null) {
                    String movieName = selectedMovie.split(" - ")[0]; // Extract movie name
                    graph.addEdge(loggedInUser, movieName);
                    unwatchedListModel.removeElement(selectedMovie);
                    watchedListModel.addElement(selectedMovie);
                    updateUnwatchedList();
                    unwatchedList.clearSelection(); // Clear selection after adding to watched
                }
            }
        });

        JPanel unwatchedPanel = new JPanel(new BorderLayout());
        unwatchedPanel.add(new JLabel("Unwatched Movies"), BorderLayout.NORTH);
        unwatchedPanel.add(new JScrollPane(unwatchedList), BorderLayout.CENTER);

        JPanel watchedPanel = new JPanel(new BorderLayout());
        watchedPanel.add(new JLabel("Watched Movies"), BorderLayout.NORTH);
        watchedPanel.add(new JScrollPane(watchedList), BorderLayout.CENTER);

        moviePanel.add(unwatchedPanel);
        moviePanel.add(watchedPanel);

        setLayout(new CardLayout());
        add(loginPanel, "Login");
        add(moviePanel, "Movies");

        switchToLoginPanel();
    }

    private void initializeGraph() {
        // Adding movies with genres
    graph.addNode("The Shawshank Redemption", "Drama");
	graph.addNode("The Dark Knight", "Action");
	graph.addNode("The Godfather", "Drama");
	graph.addNode("Fight Club", "Drama");
	graph.addNode("Inception", "Thriller");
	graph.addNode("Goodfellas", "Action");
	graph.addNode("Dune: Part Two", "Sci-Fi");
	graph.addNode("Spider-Man: Across the Spider-Verse", "Adventure");
	graph.addNode("Interstellar", "Sci-Fi");
	graph.addNode("Brokeback Mountain", "Romance");
	graph.addNode("Logan", "Action");
	graph.addNode("La La Land", "Romance");
	graph.addNode("Titanic", "Romance");
	graph.addNode("Black Panther", "Action");
	graph.addNode("Arrival", "Sci-Fi");
	graph.addNode("Blade Runner 2049", "Sci-Fi");
	graph.addNode("Saving Private Ryan", "Adventure");
	graph.addNode("Schindler's List", "Drama");
	graph.addNode("Gladiator", "Action");
	graph.addNode("Se7en", "Thriller");
	graph.addNode("The Lord Of The Rings: The Two Towers", "Fantasy");
	graph.addNode("The Lord Of The Rings: The Return Of The King", "Fantasy");
	graph.addNode("Terminator 2: Judgment Day", "Action");
	graph.addNode("2001: A Space Odyssey", "Sci-Fi");
	graph.addNode("Avengers: Endgame", "Action");
	graph.addNode("The Matrix", "Sci-Fi");
	graph.addNode("Parasite", "Thriller");
	graph.addNode("The Godfather Part II", "Action");
	graph.addNode("Mad Max: Fury Road", "Adventure");
	graph.addNode("Avengers: Infinity War", "Action");
	graph.addNode("Before Sunrise", "Romance");
	graph.addNode("The Notebook", "Romance");
	graph.addNode("Pretty Woman", "Romance");
	graph.addNode("No Hard Feelings", "Comedy");
	graph.addNode("Pride & Prejudice", "Romance");
	graph.addNode("Eternal Sunshine of the Spotless Mind", "Romance");
	graph.addNode("Friends with Benefits", "Romance");
	graph.addNode("500 Days of Summer", "Romance");
	graph.addNode("The Hangover", "Comedy");
    }

    private void switchToLoginPanel() {
        CardLayout cl = (CardLayout) getContentPane().getLayout();
        cl.show(getContentPane(), "Login");
    }

    private void switchToMoviePanel() {
        populateWatchedList();
        updateUnwatchedList();
        CardLayout cl = (CardLayout) getContentPane().getLayout();
        cl.show(getContentPane(), "Movies");
    }

    private void updateUnwatchedList() {
        unwatchedListModel.clear();
        List<String> unwatchedMovies = graph.recommendMovies(loggedInUser, List.of());
        for (String movie : unwatchedMovies) {
            String genre = graph.nodes.get(movie).genre;
            unwatchedListModel.addElement(movie + " - " + genre);
        }
    }

    private void populateWatchedList() {
        watchedListModel.clear();
        Node user = graph.nodes.get(loggedInUser);
        if (user != null) {
            for (Node movie : user.getConnections()) {
                watchedListModel.addElement(movie.name + " - " + movie.genre);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new MovieRecommendationApp().setVisible(true);
            }
        });
    }
}