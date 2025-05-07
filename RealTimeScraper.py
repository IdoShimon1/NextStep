import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Set up your Chrome WebDriver (adjust the executable path as needed)
driver = webdriver.Chrome()

def linkedin_login(driver, username, password):
    """Log into LinkedIn."""
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)  # wait for the page to load
    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").send_keys(password + Keys.RETURN)
    time.sleep(3)  # allow time for login to complete

def search_profiles(driver, job_title, max_profiles=50):
    """
    Search LinkedIn for people by job title and collect unique profile URLs.
    Note: This uses a very basic CSS selector strategy; you might need to adjust selectors.
    """
    # Build a search URL. Adjust query parameters as needed.
    search_url = f"https://www.linkedin.com/search/results/people/?keywords={job_title}"
    driver.get(search_url)
    time.sleep(3)
    
    profiles = set()
    while len(profiles) < max_profiles:
        # Find profile link elements; this CSS selector may require adjustments.
        profile_elements = driver.find_elements(By.CSS_SELECTOR, "a.app-aware-link")
        for elem in profile_elements:
            href = elem.get_attribute("href")
            if href and "linkedin.com/in/" in href:
                profiles.add(href)
                if len(profiles) >= max_profiles:
                    break
        # Scroll down to load more results
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    return list(profiles)[:max_profiles]

def scrape_profile(driver, profile_url):
    """
    Navigate to a profile URL and return the full HTML content.
    You can extend this to click on 'see more' buttons if needed.
    """
    driver.get(profile_url)
    time.sleep(3)
    
    # Optionally scroll to force dynamic sections (like skills) to load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Retrieve the entire page source (similar to doing a Ctrl+A and copying everything)
    page_content = driver.page_source
    return page_content

# --- Usage Example ---
username = "legitimatestoreido@gmail.com"
password = "Noam2012"
job_title = "Software Engineer"  # Replace with any job title from your list

# Log in to LinkedIn
linkedin_login(driver, username, password)

# Get a list of profile URLs for the given job title
profile_urls = search_profiles(driver, job_title, max_profiles=50)
print(f"Found {len(profile_urls)} profiles.")

# Scrape each profile's full page HTML (this includes all loaded sections like skills)
profiles_data = {}
for url in profile_urls:
    html_content = scrape_profile(driver, url)
    profiles_data[url] = html_content
    print(f"Scraped profile: {url}")
    
# You can now process or save profiles_data as needed

# Always remember to quit the driver when done
driver.quit()
