"""Shared style bank used by multiple diversify methods.

Each entry is a *style set* keyed by a descriptive label.  The value is a
list of example sentences that together define a target writing style.

Both TinyStyler (via authorship embeddings) and the prompting method (via
few-shot in-context examples) draw from this bank.

Select styles from it with the ``styles`` parameter, or pass your own
example texts via ``style_examples``:

    from diversify_text import diversify

    results = diversify(
        texts,
        styles=["recipe", "personal_blog"],
        style_examples={
            "scientific": [
                "The data clearly indicate a statistically significant result.",
            ],
        },
    )

Several examples are drawn from the CORE corpus (https://doi.org/10.1007/s10579-013-9256-1).
"""

# Bank order matters: n selects the first n styles, so the best-curated
# styles come first.
DEFAULT_STYLE_BANK: dict[str, list[str]] = {
    "informal_tinystyler": ["the real world, the newly weds and laguna beach", "Contact Warner Bros.or just go to ebay.I dont think youll find any", "that I'm a woman's man with no time to talk!", "When you have an eye problem so you see 3,not 2  ( :", "cant wait for a new album from him.", "I'll pick one of my favorite country ones...", "idk.....but i have faith in you lol", "Wang Chung - Everybody Have Fun Tonight", "i am gonna have to defend the werewolf here.", "YEAH, AND I WASN'T VERY COMFORTABLE WITH IT EITHER...", "IF YOU TEXT YOUR ANSWER IN IT MIGHT IF YOU DON'T HAVE TEXT MESSAGES IN YOUR PLAN", "he is about 83 yrs old", "Till they run out of ideas", "eminem because his some of his music is just so funny and relevent to todays pop music enviorment.", "Im a server for Bennagn's and LOVED IT!!", "sheet music plus is not free.", "i don't know why i'm so crazy about it.", "joining his son at the breakfast table he asked his son what happened last night whats all this for?", "bucky should, but probably somebody good will", "i like her whole blue-eyed soul style but not her", "well, it gives a nice feeling to choose and buy things.", "or did the trial drag on and on?", "and he's like 'no' so she hugs him.", "OH i think its going to be Kevin Covaias.", "are you asking if she's good at what she does?", "I dont wanna die without any scars."],
    "obama_tinystyler": ["Good afternoon, everybody.", "Let me start out by saying that I was sorely tempted to wear a tan suit today -- (laughter) -- for my last press conference.", "But Michelle, whose fashion sense is a little better than mine, tells me that's not appropriate in January.", "I covered a lot of the ground that I would want to cover in my farewell address last week.", "So I'm just going to say a couple of quick things before I start taking questions.", "First, we have been in touch with the Bush family today, after hearing about President George H.W. Bush and Barbara Bush being admitted to the hospital this morning.", "They have not only dedicated their lives to this country, they have been a constant source of friendship and support and good counsel for Michelle and me over the years.", "They are as fine a couple as we know.  And so we want to send our prayers and our love to them.  Really good people.", "Second thing I want to do is to thank all of you.", "Some of you have been covering me for a long time -- folks like Christi and Win.", "Some of you I've just gotten to know.  We have traveled the world together.", "We've hit a few singles, a few doubles together.", "I've offered advice that I thought was pretty sound, like \"don't do stupid…stuff.\"  (Laughter.)", "And even when you complained about my long answers, I just want you to know that the only reason they were long was because you asked six-part questions.  (Laughter.)", "But I have enjoyed working with all of you.", "That does not, of course, mean that I've enjoyed every story that you have filed.", "But that's the point of this relationship.", "You're not supposed to be sycophants, you're supposed to be skeptics.", "You're supposed to ask me tough questions.", "You're not supposed to be complimentary, but you're supposed to cast a critical eye on folks who hold enormous power and make sure that we are accountable to the people who sent us here.", "And you have done that.", "And you've done it, for the most part, in ways that I could appreciate for fairness even if I didn't always agree with your conclusions.", "And having you in this building has made this place work better.", "It keeps us honest.", "It makes us work harder.", "It made us think about how we are doing what we do and whether or not we're able to deliver on what's been requested by our constituents.", "And for example, every time you've asked \"why haven't you cured Ebola yet,\" or \"why is that still that hole in the Gulf,\" it has given me the ability to go back to my team and say, \"will you get this solved before the next press conference?\"  (Laughter.)"],
    "question_tinystyler": ["Where can I find a good restaurant nearby? There's a great Italian restaurant just around the corner.", "How much does this book cost? This book costs $15.", "What should I wear to the party? You should wear something casual but nice, like a sundress or a polo shirt.", "When is the next bus arriving? The next bus is arriving in 10 minutes.", "Do you know how to fix a flat tire? Yes, I can show you how to fix a flat tire.", "Why is the sky blue? The sky is blue because of the way the Earth's atmosphere scatters sunlight.", "Can you recommend a good movie? I recommend watching \"Inception\"; it's a great film.", "Where did you buy your shoes? I bought my shoes at the mall.", "How do I make a reservation at the restaurant? You can make a reservation by calling the restaurant or booking online.", "What time does the store close? The store closes at 9 PM.", "Is there a pharmacy nearby? Yes, there's a pharmacy just two blocks down the street.", "What are the symptoms of a cold? Common symptoms of a cold include a runny nose, sore throat, and cough.", "How often should I water my plants? You should water your plants once a week, or when the soil feels dry.", "Can you explain this math problem to me? Of course, let me show you how to solve it.", "Where can I park my car? You can park your car in the parking garage across the street.", "What is the best way to learn a new language? The best way to learn a new language is through consistent practice and immersion.", "Do you have any dietary restrictions? Yes, I'm allergic to nuts.", "How do I get to the nearest hospital? Take the main road north for three miles, and you'll see the hospital on your right.", "What time is your appointment? My appointment is at 3 PM.", "Can I pay with a credit card? Yes, we accept credit card payments.", "What is the Wi-Fi password? The Wi-Fi password is written on the router.", "How do I change the batteries in this remote? Open the back cover and replace the old batteries with new ones.", "Where did you go on vacation? I went to the beach in Hawaii.", "Can I borrow your book? Sure, just make sure to return it by next week.", "What time does the meeting start? The meeting starts at 10 AM.", "Do you have any pets? Yes, I have a dog and two cats."],
    "formal_stel": ["He has a very distinct walk.", "It's weight measurement is 1, the question is deceptive!", "Being intimate with one's closest friend's romantic partner?", "It is dependent on the amount they will pay.", "Santa was excessively overweight while the woman was driving.", "I love both characters, so that's a hard decision. But I think werewolf is my favorite.", "And the indication is: At the moment, the Hip Hop genre is decreasing in quality!", "Music involving a distinctive beat and smooth flow.", "I am merely indifferent to them.", "I don't know, Thunder Struck is full of guitar riffs, but I enjoy it.", "Yes, I continue to watch it in the mornings aired by TBS.", "I'm unable to recall. I misplaced my signed photo plus my photograph alongside the boys!", "	Will you leave your e-mail address so that we can communicate in the future?", "He was in extreme home makeover.", "You need to say 'Amen' to make it stop.", "Take a look at yourself in the mirror and laugh.", "I dabbed on some Armani 'Black Code'. any woman wearing, 'Design,' has me hypnotized. I would tell her, 'Anything you wish, sweetheart.'", "Three attractive women walk by and express their sympathy for him.", "Those are the only ones I can remember currently.", "In my opinion it is Maya, she is sarcastic but also down to earth.", "However, he certainly prompts me to laugh.", "It is most definitely one of my favorite shows.", "I enjoy listening to the Goo Goo Dolls and Relient K."],
    "song_lyrics": [
        "It would be a pleasure just to know just a little bit more, I could grow quite fond of your acquaintance, of this I'm sure.",
        "If you want a raise in pay, I'll tell you what to do. Go and ask the boss for it and he will give it to you.",
        "Shadows are finally freed to hover...",
        "O that I had a thousand voices and with a thousand tongues could tell of Him in Whom the earth rejoices, Who all things wisely does and well!",
        "Once I had a gal, as sweet as she could be. Once I had a gal, and she was right for me.",
    ],
    "advice": [
        "Sometimes, people just don't feel well. But if you don't feel well more than sometimes, it may be helpful to talk to someone about it.",
        "Women's Uggs have been in the fashion industry for a long time now and they have become a timeless piece to own in the fashion industry as they are an extremely comfortable brand of shoe.",
        "Martha Beck shows you what to do with a significantly annoying significant other.",
        "In Lifeclass this week, a lonely mother of one asks Lesley Garner what she should do about the fact that her husband works so hard that he doesn't have time for his family.",
        "What do you need to do to save your relationship? Often when relationships are in trouble there are certain similar issues that prevail.",
    ],
    "sports_report": [
        "The art of moaning is almost intrinsically built within us as football fans. As much as we love a 30-yard-screamer or a derby day thumping, we also love to indulge in a spot of match-day whining.",
        "The 25-year-old did have chances at Anfield.",
        "Welcome, one and all, to TheCheckingLine.com, your source for unique hockey insight.",
        "Illinois coach Tim Beckman already meets regularly with the offensive staff, but after the Illini's recent struggles he says he needs to spend more time with them.",
        "Two losses to both of the Manchester clubs may well have been expected for most Villa fans, but a place in the bottom three was not.",
    ],
    "opinion_blog": [
        "The art of moaning is almost intrinsically built within us as football fans.",
        "Without Collaboration, There is No Innovation! I recently wrote about the importance of Innovation.",
        "Making the right hire is one of the hardest tasks that businesses ask HR to help perform.",
        "However much the debate over irregular boat arrivals has been refocused by the shocking loss of life at sea, it is plain that domestic politics continue to motivate the main players.",
        "I can't remember when I last saw a new guitar band promoted on TV or releasing a professionally shot video.",
    ],
    "news_report_blog": [
        "A transportation advocacy group is circulating a list of 100 questions aimed at broadening the British Columbia government's consultation on coastal ferry services.",
        "The value of NYC housing construction starts more than doubled to $1.9 billion in the first half of 2012, hitting a four-year high.",
        "Although official inflation in Argentina ranges 11.5%, private estimates from advisory economic and financial groups as well as market operators support estimates above 25%.",
        "This week's Newsweek declared Obama 'the first gay president.'",
        "As the 25th edition of IDFA kicks off in Amsterdam today, realscreen examines the exceptional recent crop of films focusing on the Israel/Palestine conflict.",
    ],
    "discussion_forum": [
        "I've been recording and mixing music for about 4 years now. My clients are always happy and they do come back but I am not even close to being happy.",
        "If this is your first visit, be sure to check out the FAQ by clicking the link above. You may have to register before you can post.",
        "I don't know about any of you, but lately I've been hearing some tasty acoustic guitar work used as background in many TV commercials.",
        "any one know how to install win 95 i know this is windows me help but there both crappy OS and theres no windows 95 help anyways.",
        "Hi, just started playing this one and I have a cow shed and two chicken houses, along with my planting field. Was wondering, do we ever get the chance to have another planting field?",
    ],
    "historical_article": [
        "The Middle Ages were not by any means ignorant of science, but its study was shaped by a very different set of assumptions than those we hold today.",
        "Natural history is situated both before and after language; it decomposes the language of everyday life, but in order to recompose it.",
        "The most indubitable feature of a revolution is the direct interference of the masses in historical events.",
        "In November 2009 Braunton and District Museum won Heritage Lottery funding to tell the story of how this coastline played an important role in military history.",
        "The following sketch of the seamy side of life as seen in Glasgow fifty years ago is from the pen of a contemporary observer.",
    ],
    "personal_blog": [
        "Its been fun to see all of the posts and the difference in the work. In the middle of all of this, Sandy happened, and Gilles Peress made a detour to cover the aftermath.",
        "As long as the three of you are all healthy and happy, that's the main thing. Cherish every moment because kids grow up so fast and before you know it she will be a teenager!",
        "The Ladies and Men that I sit and work with are completely lovely. I didn't draw them all in this picture.",
        "As my family well knows, there comes a day in November (December is just too late) when I do nothing but bake cranberry bread.",
        "How glad am I that I didn't start the ten days of hormones? As yesterday moved on I started to feel worse and worse and by late last night I was pretty miserable.",
    ],
    "reviews": [
        "From Creative Labs to Tritton, get your head in the game with crystal clear comms.",
        "Canada's first purpose-built opera house was completed in 2006 and is the home of the Canadian Opera Company and the National Ballet of Canada.",
        "Who killed Martin Luther King Jr. on April 4, 1968, while he was standing on a balcony at the Lorraine Motel in Memphis, Tennessee?",
        "As an avid Disney fan as a child (and now as well), I went to the screening of Wreck-It Ralph with high expectations.",
        "As an unabashed fan of Garcia Marquez, I've been rationing the unread stories so they can last me a while longer, but I couldn't resist.",
    ],
    "religious_blogs_sermons": [
        "We live in a world that appears to have lost its moral moorings.",
        "In the Name of God, the Lord of overpowering majesty, the All-Compelling. Hallowed be the Lord in Whose hand is the source of dominion.",
        "And which of you by worrying can add a single hour to his life's span?",
        "It's not every day that journalists are presented with a style guide to interpret the most enduring international bestseller on the planet.",
        "The US government is funding an unusual national training program to help police deal with volatile confrontations involving highly trained veterans.",
    ],
    "research_article": [
        "The Buenos Ayres Notebook takes its name from the city Buenos Aires, on the southern shore of the Rio de la Plata, on the coast of Argentina.",
        "Research co-authored by AUT University's head of hospitality suggests holiday-makers, especially women, should think twice about their safety when boarding a cruise ship.",
        "This is an Open Access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium.",
        "Equalising constituency sizes is likely to reduce the electoral bias in favour of Labour but only minimally so.",
        "Astronomers using ESO's Very Large Telescope have identified a body that is very probably a planet wandering through space without a parent star.",
    ],
    "travel_blog": [
        "The wholesale Group transported Sydney agents into Asia and India last night, tantalising the senses with local sounds, sights and objects.",
        "If you've already been to South-East Asia, you won't need me to tell you how great it is.",
        "Studs and leather replace Speedos in the real performance. Actors change in and out of neoprene costumes in hidden underwater changing rooms.",
        "Get ready for Christmas with our Christmas Wreath making workshop.",
        "The architectural glory of European countries is represented by their magnificent castles.",
    ],
    "how_to": [
        "We said that classical training was interested mainly in three things: training the horse to carry a rider, training the horse to be obedient, and improving the horse's athletic ability.",
        "Public speaking is something that most of us have to do at some time in our lives, but most of us would rather not do. This experience starts in school.",
        "After days, weeks or months of pondering, consideration and research, you're ready to test drive that new car or truck.",
        "After years of poor posture, 5'2\" writer Coco Myers learns she's coming up short.",
        "Change your oil regularly. If you don't change your oil often enough your car will start to lose valuable lubrication and the result is damage to your engine.",
    ],
    "description_of_a_thing": [
        "Sharing photos or videos shot in class during the learning process.",
        "400 Years of Seasonal Traditions in English Homes.",
        "For most lawyers, attaining partnership is an achievement, a moment of happiness and relief.",
        "We have recorded our new release entitled BLACK LIGHTNING and need your help to get this killer record out to the fans!",
        "Hull is the most affordable place to live in the UK. New research suggests that 98 per cent of homes in Hull are within financial reach for first-time buyers on average local incomes.",
    ],
    "other_information": [
        "At times, the blog may not be accessible, or may be slow. We will inform our users once the blog resumes its normal functions. We appreciate your patience in this matter.",
        "There is almost no bigger story in public library service than the challenges of getting eContent for patrons.",
        "The element of a person that enables them to be aware of the world and their experiences, to think, and to feel; the faculty of consciousness and thought.",
        "The annual meeting of shareholders of Auckland International Airport Limited was held at the TelstraClear Pacific Events Centre today.",
        "Where it is feasible, a syllabus will be released, as is being done in connection with this case, at the time the opinion is issued.",
    ],
    "prayer": [
        "In the Name of God, the Lord of overpowering majesty, the All-Compelling. Hallowed be the Lord in Whose hand is the source of dominion.",
        "These coats of skin had a significancy. The beasts whose skins they were must be slain, slain before their eyes to shew them what death is.",
        "An online Bible Study course about the life of Jesus. This book is in EasyEnglish Level B.",
        "Here are the resources you need for this Sunday. Please feel free to choose what is suitable for your congregation's needs.",
        "For all the saints who went before us, who have spoken to our hearts and touched us with your fire, we give you thanks, O God.",
    ],
    "description_with_intent_to_sell": [
        "We have recorded our new release entitled BLACK LIGHTNING and need your help to get this killer record out to the fans!",
        "An Acadian-style cabin constructed completely of rough sawed Southern Yellow Pine, surrounded by split rail fence.",
        "As a Smartline Personal Mortgage Adviser, I am so much more than a home loan processor.",
        "Your credit card will NOT be charged when you start your free trial or if you cancel during the trial period.",
        "When Lisa said that I could choose any toy I wanted to review I was thrilled!",
    ],
    "description_of_a_person": [
        "Mary Ann Sieghart has been writing about politics since the mid-1980s. After stints at the FT and Today newspaper, she joined The Economist in 1986.",
        "Born in 1707 to a country parson in southern Sweden, the young Linnaeus showed a keen interest in plants and flowers.",
        "Of our chosen five young male actors, none attended full-time drama school.",
        "\"She does a lot of hard work around the school,\" said Ashley Ramkissoon, a 13-year-old eighth-grader.",
        "Robert Petrancosta's first truck drive happened by chance and he could never have imagined it would shape his future.",
    ],
    "magazine_article": [
        "In early 2008, David Kinch, the man behind two-Michelin-starred Manresa in Los Gatos, agreed to appear on Iron Chef America.",
        "We're all a little bored with all this rivalry between The Wanted and One Direction and all their fans.",
        "A recent survey suggests that in many schools the Pupil Premium is doing little to meet the educational needs of children from low-income families.",
        "At Stylenoir Magazine we have been obsessed by the K-Pop girl group 2NE1 since we ever first set eyes on them.",
        "There are a lot of unknowns, notes Bret Drake, one of the architects for NASA's Mars missions. He's optimistic that people can be safely landed on Mars, but it won't happen soon.",
    ],
    "faq_about_how_to": [
        "Why am I never called to be polled? You have roughly the same chance of being called as anyone else living in the United States who has a telephone.",
        "Where do you get your industry benchmarks? We develop benchmarks from a number of sources.",
        "A mother doe will leave her fawn in what she thinks is a safe place while she heads off to find food.",
        "Will all previously filed labor certification applications be converted and processed under PERM?",
        "Ask you how you have been feeling, if you cough a lot or if you find it hard to breathe sometimes.",
    ],
    "encyclopedia_article": [
        "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Despite their families' financial constraints, both parents attended the University of Oxford.",
        "Desclos did not reveal herself as the author for forty years after the initial publication.",
        "Puzzled by the geographical distribution of wildlife and fossils he collected on the voyage, Darwin began detailed investigations and in 1838 conceived his theory of natural selection.",
        "On 3rd August, 1914, German troops crossed the Belgian border in the narrow gap between Holland and France.",
        "The curveball is a type of pitch in baseball thrown with a characteristic grip and hand movement that imparts forward spin to the ball.",
    ],
    "short_story": [
        "The sun was setting on the night of our birthday. My twin and I turned twelve years old that day -- practically grown-ups.",
        "The Charmed Ones mistrust their new whitelighter from the future.",
        "One morning, as Miss Matty and I sat at our work, Miss Matty had not changed the cap with yellow ribbons that had been Miss Jenkyns's best.",
        "We were engaged after the first walk that we took. He was a cashier in an office in Leadenhall Street.",
        "Colonel Cargill was a forceful, ruddy man. Before the war, he had been an alert, hard-hitting, aggressive marketing executive. He was a very bad marketing executive.",
    ],
    "interview": [
        "From video games to comics, many artists find themselves tasked with finding ways to evolve an established franchise, which can be particularly tricky.",
        "This is Karl Moore of the Desautels Faculty of Management at McGill University, Talking Management for The Globe and Mail.",
        "Donnington Valley boasts amongst its assets a wonderful golf course and a hotel collection that includes a vineyard in California.",
        "At the end of September, the 3% Conference happened for the first time in San Francisco.",
        "I met up with iconic British designer Paul Smith at London Fashion Week ahead of his One Night Only tour in Australia.",
    ],
    "information_blog": [
        "Thanks to community websites like meetup.com, finding your tribe in the urban jungle has never been easier.",
        "Why is it important to be green in the home? What are the different ways that you can make your home more environmentally friendly?",
        "Clever design can make the best use of space, even in smaller houses. We suggest some ways to optimise the areas in your home.",
        "The percentage of people who changed residences between 2010 and 2011 reached a record low.",
        "Logging in the Vanimo Green River region of Papua New Guinea has shown a complete disregard for the rights of landowners.",
    ],
    "reader_viewer_responses": [
        "I can't remember when I last saw a new guitar band promoted on TV or releasing a professionally shot video.",
        "My partner and I stayed mid week for one night and paid $149 for a deluxe room with breakfast included.",
        "Post your journal entry about your experience exploring Second Life here and any ideas about how this landscape may lend itself to learning.",
        "With loads of money, you might have all the people you used to know come out of the woodwork guilting you into giving them money.",
        "We stayed here with our 2 children. Lovely hotel, VERY clean and the food was fine, the children's entertainment was fab.",
    ],
    "letter_to_editor": [
        "\"If people want to read lots of gushing rubbish,\" writes reader Anne Loader, \"there are plenty of sources for this.\"",
        "Millennium today features a special guest post responding to a review of 'Religious Politics and Secular States: Egypt, India and the United States.'",
        "Sorry, we are having technical issues. Please try again later during the day.",
        "A few years ago, at Palm Island, there were riots, buildings burnt, following the death in custody of an Aboriginal man.",
        "I'm a 22-year-old girl and have never worked in my life. I left school five years ago.",
    ],
    "question_answer_forum": [
        "I will wait out December 2012, just because I want to be amused when nothing happens and the end of the world doesn't come.",
        "I just have a few questions about the footage: Does this take place in the second movie?",
        "How can I tell when a movie is going to be removed? I saw a movie I wanted to watch last week but now it's gone.",
        "What was one of William the Conqueror's major accomplishments as King of England? William initiated many major changes.",
        "If there is a god, why is there suffering in the world? With the assumption that god is good, why is there suffering?",
    ],
    "faq_about_information": [
        "You need to complete a teacher education program at a Canadian university and qualify for a teaching certificate in the province where you complete it.",
        "How can I tell when a movie is going to be removed? I saw a movie I wanted to watch last week but now it's gone.",
        "We'll work to give you the best information we can find to help you find out everything you need to know about how old you have to be to work.",
        "Are shovels ergonomically designed? The design of shovels and spades did not come from an ergonomist's drawing board. They are basic tools that have evolved over many centuries.",
        "Why does your cat wash your hair or face? The first thing a kitten experiences, even before it can see, is its mother licking and washing.",
    ],
    "formal_speech": [
        "Once upon a time, long long ago, I learned how to reduce a fraction to its lowest terms.",
        "This is the opening speech crafted by the Unit 4 Bargaining Team and delivered on October 12, 2011.",
        "Thank you for inviting me to speak today at the Annual General Meeting of the Settlement Council of Australia. It is a pleasure to be here.",
        "I am delighted to be in Townsville to deliver the Dalton Memorial Address.",
        "I was elected by the Australian people as Prime Minister of this country to bring back a fair go for all Australians.",
    ],
    "persuasive_article_or_essay": [
        "When anyone talks about email marketing, you will almost always hear the phrase: the money is in the list.",
        "There are an estimated 12.3 million people globally who have been stripped of their dignity through human trafficking.",
        "Sports have the ability to help your mood, keep your body in shape, relieve stress, build confidence, and they help build relationships.",
        "Apple is expected to unveil a new, smaller form factor iPad tablet at an event in San Francisco later on Tuesday.",
        "This post examines a particularly horrendous violation of children's rights: the phenomenon of child soldiers.",
    ],
    "course_materials": [
        "It takes two to speak the truth -- one to speak and the other to hear.",
        "To see the answers, highlight the space directly below the question by holding down the mouse button and dragging across.",
        "By approaching writing as a process, instructors encourage students to avoid closure on a piece of writing before exploring its full possibilities.",
        "Using your plant parts and functions information sheets, create a game to help someone learn about plant parts and their functions.",
        "Beyond Stimulus discusses a response to the aftermath of the economic and financial crisis that goes further than fiscal and monetary stimulus.",
    ],
    "legal_terms_and_conditions": [
        "The victim personal statement scheme gives victims an opportunity to describe the wider effects of the crime upon them and express their concerns.",
        "The police can come up to you and ask you questions at any time, but this does not mean you are obliged to answer all of their questions.",
        "Your agency must select the method most advantageous to the Government, when cost and other factors are considered.",
        "\"Commissioner\" means the Government Film Commissioner appointed under section 16.",
        "Where it is feasible, a syllabus will be released, as is being done in connection with this case, at the time the opinion is issued.",
    ],
    "recipe": [
        "Parathas are an Indian unleavened bread and an integral part of many Indian meals. Any subzi or curry dish can be fabulous when accompanied with fresh, hot parathas.",
        "They've got a great American 1948 Ford Pickup and their speciality is the Japanese dumpling called Gyoza -- often called pot stickers in America.",
        "This is one dish I always look for at markets and food-type fairs and it never lets me down -- the big large pots they cook it in just look so inviting!",
        "Cut a peeled brown onion lengthways into 5mm-thick slices. Place one large onion slice in the centre of the pastry.",
        "Ahoy there me mateys! We have just celebrated my youngest son's 5th birthday. He decided that he wanted a pirate party this year.",
    ],
    "poem": [
        "It was kind of awkward, so he took it away. \"This was, like, the coolest place in the world when I was a kid.\" She looked around.",
        "Dear Guido Fawkes, please do us a favour, give us all something that we can savour. Come back from the dead if you possibly can.",
        "Out there, my old life tempts, a voice cries, \"Fail!\", and tells me there are better things to do. The world shrinks down, we both exhale, and drift together, touching souls, we two.",
        "I am lucky that John is on the staff at Manchester. I discovered his poetry this way. He has a lovely lyrical voice with sharp edges.",
        "Your poem, here it halts in English. And they say the language speaks you.",
    ],
    "editorial": [
        "First of all, I want to apologise to Eurogamer's readers for not saying anything else about why I edited Rab Florence's column last week until now.",
        "In recent weeks our newspapers and television screens have been dominated by tragic stories of NHS patients being denied access to drugs.",
        "To be a car enthusiast in Japan takes a lot more effort than in America or the UK. First of all there are hardly any parking spaces.",
        "Journal editors are expressing enthusiastic support for a new open access option announced recently by SPIE, the international society for optics and photonics.",
        "The election of Barack Obama has had surprisingly little impact on a nation fixated with race.",
    ],
    "transcript_of_video_audio": [
        "I'm a golf pro. So obviously I wake up in the garbage.",
        "I used to come from the world of supermarkets where choice, too much choice, was overwhelming.",
        "It's great to be here in Queensland. I'm joined by the Deputy Prime Minister, Wayne Swan.",
        "Good afternoon, ladies and gentlemen. The end of the Kokoda campaign, 1 November 1942.",
        "With President Obama scoring a decisive re-election, journalist John Nichols argues that the 2012 election represented a significant shift.",
    ],
    "technical_report": [
        "If you are using any of the book options that are built into SmugMug, I think you can just use the cart functionality as a better option.",
        "Qantas Group today announced Underlying Profit Before Tax of $95 million for the year ended 30 June 2012.",
        "While it's never a good idea to bloat your webpage with too much unneeded content, there are times when you can't avoid it.",
        "The national security depends on our defense installations and facilities being in the right place, at the right time, with the right qualities.",
        "Most recent versions of the operating system will also not accept the password if it does not contain a non-alphabetic character.",
    ],
    "advertisement": [
        "Win yourself a FREE copy of the BradyGames official strategy guide for the recently released Resident Evil 6.",
        "Speak softly. Train hard. Yelling is for those that can't get results any other way.",
        "Hotel Alte City is a cosy hotel decorated in the style of old Berlin with many antiques on view. Low price with high comfort in the City!",
        "It marks the first day of autumn, and it seems that Mother Nature has an uncanny sense of timing with the weather.",
        "Do you ever sit at home in front of your computer connecting with someone over twitter while discussing a common interest like cooking or music?",
    ],
    "technical_support": [
        "If your computer is unable to connect to Windows Phone Update, try the following suggestions.",
        "The only supported way to modify MAPI profiles programmatically is through Extended MAPI.",
        "I got WAMP working, everything seems ok, but the back office service browser can't find any service I place in the service folder.",
        "I've recently upgraded from Sky's ADSL2+ product to their Fibre to the Cabinet internet package.",
        "Do you disable Autorun on your PC? Many viruses execute malicious code by running automatically via a thumb drive or network drive.",
    ],
    "other_informational_persuasion": [
        "5 Great Reasons to Optimize Your Landing Pages. Landing page optimization allows you to improve your user experience.",
        "You may have heard about Heather Mallick's column denouncing the Better Halves for contributing to First Place, a crisis pregnancy centre.",
        "If you read the Fix regularly, you won't be surprised that the team is getting pretty excited about the upcoming Republican and Democratic National Conventions.",
        "The Earth's average temperature has warmed by about 0.76C over the past 100 years, with most of this warming occurring in the past 20 years.",
        "You will want to see the greatest threat presented by the extensive number of politicians who have faced criminal charges.",
    ],
    "other_spoken": [
        "The only thing they have to look forward to is hope. And you have to give them hope.",
        "Because almost every interesting thing we say was probably said by someone else first. Life is beautiful, as long as it consumes you.",
        "Suppose neutral angels were able to talk God and Satan into settling out of court. What would be the terms of the compromise?",
        "Transparency has to go with the territory. I think you and I both know that nothing will ever be revealed until after the fact.",
        "We've all heard that we have to learn from our mistakes, but I think it's more important to learn from successes.",
    ],
    "other_opinion": [
        "Jane-Michele Clark is president of The Q Group, a strategic positioning and marketing firm with a 30 year history.",
        "We are listening -- we want to hear from you! What is up with the media continually trying to shove this trash down our throats?",
        "There are few places in the world as beautiful as the area surrounding the home of Angela and Alfonso Imperato.",
        "The South Australian Government's now-abandoned plan to ban planking wasn't just a dumb idea. It was a breathtakingly stupid waste of time.",
        "This Thanksgiving I am thankful for accessible healthcare.",
    ],
    "tv_movie_script": [
        "A host of characters approach the challenge with various techniques and opinions, culminating into a debacle that no-one is prepared for.",
        "It has fallen to us, to defend Jerusalem, and we have made our preparations as well as they can be made. None of us took this city from Muslims.",
        "Ray, people will come. They'll come to Iowa for reasons they can't even fathom. They'll turn up your driveway, not knowing for sure why they're doing it.",
        "There's a reason you separate military and the police. One fights the enemies of the state, the other serves and protects the people.",
        "What you're telling me Sir, and correct me if I'm wrong, is that the infantry attack on Lone Pine and our Light Horse attack on the Nek are diversions.",
    ],
    "other_narrative": [
        "U.S. Supreme Court justice Stephen Breyer told reporters this week that he cannot stop himself from deliberating over even the most mundane aspects of his personal life.",
        "But there was none like unto Ahab, which did sell himself to work wickedness in the sight of the Lord.",
        "It's underway! This morning, at roughly 10 a.m. Pacific time, we began rolling out the Mango update to phones around the world.",
        "This podcast was recorded live as part of the Pub History Society conference on the 20th February 2010 at The National Archives.",
        "Leah is growing even more concerned about Jamie when she starts to notice things around her house missing or misplaced.",
    ],
    "other_forum": [
        "Hi Robert, or should I say Graham. I have found that it is quite often that people use their second name especially in census returns.",
        "Drums, bass, guitar looking for FEMALE SINGER and KEYBOARDIST for fun covers band. To play as much as possible for decent fees.",
        "Would not support what he was pushing in the open forums. Redirecting members to his site. Complaints from members.",
        "We've ordered 8 machines now and are happy with all of them. Just to let you know that the printer arrived safely and is exactly what was promised.",
        "I have found that it is quite often that people use their second name, which has caused me some problems in genealogical research.",
    ],
}


