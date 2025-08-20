import spacy
import re
from typing import List, Dict, Set, Tuple, Optional

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_sm")


class SentenceWorkflowExtractor:
    """
    Extracts meaningful workflow tasks from sentences with context preservation.

    This class processes natural language text to identify business workflow elements
    including tasks, gateways, timer events, and message events. It maintains context
    across sentences to improve extraction accuracy and provides intelligent sequencing
    of identified workflow elements.
    """

    def __init__(self):
        """
        Initialize the extractor with predefined patterns and vocabularies.

        Sets up business verb vocabularies, decision patterns, timer patterns,
        message patterns, and transition indicators used for workflow element
        identification and sequencing.
        """
        # Business action vocabulary for task identification
        self.business_verbs = {
            'submit', 'review', 'approve', 'reject', 'assign', 'conduct', 'perform',
            'evaluate', 'assess', 'analyze', 'validate', 'test', 'develop', 'create',
            'build', 'design', 'implement', 'execute', 'monitor', 'track', 'report',
            'compile', 'generate', 'issue', 'authorize', 'negotiate', 'source'
        }

        # Regular expressions for identifying decision logic in text
        self.decision_patterns = [
            r'if\s+([^,]+),\s*([^.]+)',
            r'when\s+([^,]+),\s*([^.]+)',
            r'should\s+([^,]+),\s*([^.]+)',
            r'approve.*?([^,]+),\s*([^.]+)',
            r'reject.*?([^,]+),\s*([^.]+)'
        ]

        # Patterns for detecting timer events and temporal constraints
        self.timer_patterns = [
            r'\b(\d+)\s*(minutes?|hours?|days?|weeks?|months?)\b',
            r'\bwithin\s+(\d+)\s*(minutes?|hours?|days?)\b',
            r'\btimer\b', r'\bdeadline\b', r'\btimeout\b', r'\breminder\b',
            r'\binactivity\b.*?(period|time)', r'\bSLA\b.*?(breach|deadline)'
        ]

        # Patterns for identifying message events and communication flows
        self.message_patterns = [
            r'\bmessage\s+event\b', r'\bnotification\b', r'\be-?mail\b', r'\bSMS\b',
            r'\bsend\s+\w+\s+(message|notification|alert)',
            r'\breceive[ds]?\s+\w*\s*(message|signal|notification)',
            r'\bcallback\b', r'\bconfirmation\b'
        ]

        # Transition words indicating strong sequential connections
        self.strong_transitions = [
            'then', 'after which', 'next,', 'subsequently', 'which triggers',
            'leading to', 'proceeds to', 'flows to', 'routes to'
        ]

    def extract_sentence_workflows(self, section_text: str, section_header: str) -> List[Dict]:
        """
        Extract workflow tasks from sentences while preserving context.

        Processes text section sentence by sentence to identify workflow elements.
        Maintains context across sentences to improve actor resolution and task
        identification. Applies intelligent sequencing to the extracted elements.

        Args:
            section_text: The text content to process for workflow extraction
            section_header: The section title for context and organization

        Returns:
            List of workflow task dictionaries with enhanced sequencing information
        """
        doc = nlp(section_text)

        sentence_context = []
        workflow_tasks = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 5:  # Skip very short sentences
                continue

            # Extract workflow elements using enhanced detection methods
            extracted_tasks = self._extract_from_sentence_enhanced(sent, section_header, sentence_context)

            # Update context for next sentence processing
            sentence_context.append(sent_text)
            if len(sentence_context) > 3:  # Keep only last 3 sentences for context
                sentence_context.pop(0)

            # Filter and add meaningful tasks
            for task in extracted_tasks:
                if self._is_meaningful_task(task):
                    workflow_tasks.append(task)

        # Apply intelligent sequencing and merging logic
        return self._apply_intelligent_sequencing(workflow_tasks)

    def _extract_from_sentence_enhanced(self, sent, section_header: str, context: List[str]) -> List[Dict]:
        """
        Enhanced extraction with prioritized element detection.

        Processes a single sentence to identify workflow elements using a priority-based
        approach: timer events first, then message events, then decision gateways,
        and finally regular tasks.

        Args:
            sent: spaCy sentence object for linguistic analysis
            section_header: Section context for element organization
            context: Previous sentences for actor resolution

        Returns:
            List of extracted workflow elements with metadata
        """
        sent_text = sent.text.strip()
        tasks = []

        # Priority 1: Timer events take precedence due to their specificity
        if self._contains_timer_event(sent_text):
            timer_task = self._extract_timer_event(sent, section_header, context)
            if timer_task:
                tasks.append(timer_task)
                return tasks

        # Priority 2: Message events for communication flows
        if self._contains_message_event(sent_text):
            message_task = self._extract_message_event(sent, section_header, context)
            if message_task:
                tasks.append(message_task)
                return tasks

        # Priority 3: Decision gateways for control flow branching
        if self._contains_decision(sent_text):
            gateway_task = self._extract_gateway(sent, section_header, context)
            if gateway_task:
                tasks.append(gateway_task)
        else:
            # Priority 4: Regular business tasks with enhanced actor identification
            regular_tasks = self._extract_regular_tasks_enhanced(sent, section_header, context)
            tasks.extend(regular_tasks)

        return tasks

    def _apply_intelligent_sequencing(self, workflow_tasks: List[Dict]) -> List[Dict]:
        """
        Apply intelligent sequencing and merge logical triplets.

        Enhances the extracted workflow tasks by merging related message-decision
        triplets and adding sequence information based on transition analysis.
        Calculates connection confidence and identifies explicit transitions.

        Args:
            workflow_tasks: List of extracted workflow tasks

        Returns:
            List of sequenced workflow tasks with connection metadata
        """
        if len(workflow_tasks) <= 1:
            return workflow_tasks

        print(f"INTELLIGENT SEQUENCING: Processing {len(workflow_tasks)} tasks")

        # Step 1: Merge related message-decision triplets
        merged_tasks = self._merge_message_decision_triplets(workflow_tasks)

        if len(merged_tasks) < len(workflow_tasks):
            print(f"Triplet merging: {len(workflow_tasks)} → {len(merged_tasks)} tasks")

        # Step 2: Add sequence information based on transition analysis
        for i, task in enumerate(merged_tasks):
            source_text = task.get('source_sentence', task.get('description', ''))

            # Analyze transition strength for sequence confidence
            has_strong_transition = any(trans in source_text.lower() for trans in self.strong_transitions)

            # Set sequence confidence based on transition analysis
            if has_strong_transition:
                task['link_confidence'] = 0.9
                task['connection_type'] = 'explicit_transition'
            else:
                task['link_confidence'] = 0.5
                task['connection_type'] = 'document_order'

            # Set next step reference for BPMN flow generation
            if i + 1 < len(merged_tasks):
                next_task = merged_tasks[i + 1]
                task['next_step_id'] = f"Step_{i + 2}"
            else:
                task['next_step_id'] = 'END'

        print(f" Sequencing complete: {len(merged_tasks)} final tasks")
        return merged_tasks

    def _extract_gateway(self, sent, section_header: str, context: List[str]) -> Optional[Dict]:
        """
        Extract decision gateway from sentence.

        Identifies decision points in the text and extracts the decision maker,
        conditions, and potential outcomes. Creates a gateway structure suitable
        for BPMN representation.

        Args:
            sent: spaCy sentence object containing decision logic
            section_header: Section context for organization
            context: Previous sentences for decision maker resolution

        Returns:
            Gateway dictionary with decision branches or None if no valid gateway
        """
        sent_text = sent.text.strip()

        # Identify the decision maker using context analysis
        decision_maker = self._find_decision_maker(sent_text, context)

        # Extract decision conditions and outcomes using pattern matching
        branches = []
        for pattern in self.decision_patterns:
            matches = re.finditer(pattern, sent_text, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip()
                outcome = match.group(2).strip()

                branches.append({
                    'condition': condition.capitalize(),
                    'outcome': outcome.capitalize()
                })

        if not branches:
            return None

        gateway_name = self._create_gateway_name(sent_text)

        return {
            'type': 'gateway',
            'actor': decision_maker,
            'name': gateway_name,
            'description': f"Decision point based on: {sent_text[:80]}...",
            'branches': branches,
            'section': section_header,
            'confidence': 0.8
        }

    def _extract_business_actors(self, sent_text: str) -> List[str]:
        """
        Extract business actors from sentence text.

        Uses named entity recognition and pattern matching to identify
        organizational entities and business roles mentioned in the text.

        Args:
            sent_text: Text to analyze for actor mentions

        Returns:
            List of identified business actor names
        """
        doc = nlp(sent_text)
        actors = []

        # Extract organizational entities using named entity recognition
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON'] and self._is_business_actor(ent.text):
                actors.append(ent.text)

        # Extract business roles using pattern matching
        role_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)\b',  # Three-word business units
            r'\b([A-Z][a-z]+ Team)\b',  # Team designations
            r'\b([A-Z][a-z]+ Committee)\b',  # Committee designations
            r'\b([A-Z]{2,4})\b'  # Business acronyms
        ]

        for pattern in role_patterns:
            matches = re.findall(pattern, sent_text)
            for match in matches:
                if self._is_business_actor(match) and match not in actors:
                    actors.append(match)

        return actors

    def _extract_business_actions(self, sent) -> List[str]:
        """
        Extract meaningful business actions from sentence.

        Identifies business verbs and their associated objects to create
        meaningful action phrases for task naming.

        Args:
            sent: spaCy sentence object for linguistic analysis

        Returns:
            List of business action phrases
        """
        actions = []

        for token in sent:
            if token.pos_ == 'VERB' and token.lemma_.lower() in self.business_verbs:
                # Build action phrase with verb and direct objects
                action_phrase = [token.lemma_.lower()]

                # Add direct objects for context
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj'] and len(child.text) > 2:
                        action_phrase.append(child.text.lower())

                action = ' '.join(action_phrase)
                if len(action) > 3:  # Filter out very short actions
                    actions.append(action)

        return actions

    def _create_task_name(self, action: str, sent_text: str) -> str:
        """
        Create meaningful task name from action and context.

        Enhances generic actions with contextual information to create
        more descriptive and meaningful task names.

        Args:
            action: The primary business action identified
            sent_text: Full sentence text for context extraction

        Returns:
            Enhanced task name with contextual information
        """
        action_clean = action.strip().capitalize()

        # Enhance generic actions with contextual objects
        if action_clean.lower() in ['review', 'approve', 'conduct', 'perform', 'analyze']:
            # Extract object being acted upon for context
            words = sent_text.split()
            for i, word in enumerate(words):
                if action.lower() in word.lower():
                    # Look for meaningful words following the action
                    next_words = words[i + 1:i + 3]
                    context_words = [w for w in next_words if w.lower() not in ['the', 'a', 'an', 'and', 'or']]
                    if context_words:
                        return f"{action_clean} {' '.join(context_words)}"

        return action_clean

    def _is_meaningful_task(self, task: Dict) -> bool:
        """
        Check if task represents meaningful business activity.

        Validates that extracted tasks meet quality thresholds for actor
        identification, action specificity, and confidence scores.

        Args:
            task: Task dictionary to validate

        Returns:
            True if task meets meaningfulness criteria
        """
        # Must have clear actor and action identification
        if not task.get('actor') or not task.get('name'):
            return False

        # Must meet minimum confidence threshold
        if task.get('confidence', 0) < 0.5:
            return False

        # Must not be overly generic single-word actions
        if len(task['name'].split()) < 2 and task['name'].lower() in ['execute', 'perform', 'do']:
            return False

        return True

    def _contains_decision(self, sent_text: str) -> bool:
        """
        Check if sentence contains decision logic.

        Uses keyword matching to identify sentences that contain
        decision-making or conditional logic elements.

        Args:
            sent_text: Text to analyze for decision indicators

        Returns:
            True if sentence contains decision logic
        """
        decision_keywords = ['if', 'when', 'should', 'approve', 'reject', 'decide', 'review']
        return any(f" {keyword} " in f" {sent_text.lower()} " for keyword in decision_keywords)

    def _is_business_actor(self, actor: str) -> bool:
        """
        Check if string represents a business actor.

        Validates actor candidates against business role patterns and
        filters out generic terms and technical artifacts.

        Args:
            actor: String to validate as business actor

        Returns:
            True if string represents a valid business actor
        """
        actor_lower = actor.lower().strip()

        # Filter out generic terms and technical artifacts
        generic_terms = {'process', 'document', 'system', 'it', 'this', 'that', 'method', 'approach'}
        if actor_lower in generic_terms:
            return False

        # Minimum length requirement
        if len(actor_lower) < 2:
            return False

        # Check for business role patterns
        business_patterns = [
            r'team$', r'committee$', r'manager$', r'lead$', r'department$',
            r'^[a-z&]{2,4}$',  # Business acronyms
            r'engineering', r'marketing', r'finance', r'supply', r'strategy'
        ]

        for pattern in business_patterns:
            if re.search(pattern, actor_lower):
                return True

        return len(actor.split()) >= 2  # Multi-word actors are typically business roles

    def _find_decision_maker(self, sent_text: str, context: List[str]) -> str:
        """
        Find who makes the decision in the sentence.

        Searches current sentence and recent context to identify the
        entity responsible for making the decision.

        Args:
            sent_text: Current sentence containing decision logic
            context: Previous sentences for actor resolution

        Returns:
            Name of the decision-making entity
        """
        # Look for explicit decision makers in current sentence
        actors = self._extract_business_actors(sent_text)
        if actors:
            return actors[0]

        # Search recent context for decision makers
        for sent in reversed(context[-2:]):  # Check last 2 sentences
            actors = self._extract_business_actors(sent)
            if actors:
                return actors[0]

        return "Decision Maker"

    def _create_gateway_name(self, sent_text: str) -> str:
        """
        Create meaningful gateway name from sentence content.

        Identifies the key decision topic to create descriptive
        gateway names for BPMN representation.

        Args:
            sent_text: Sentence containing decision logic

        Returns:
            Descriptive gateway name
        """
        # Extract key decision words for gateway naming
        decision_words = ['approve', 'reject', 'review', 'assess', 'evaluate', 'screen']
        for word in decision_words:
            if word in sent_text.lower():
                return f"{word.capitalize()} Decision"

        return "Decision Point"

    def _create_task_description(self, sent_text: str) -> str:
        """
        Create task description from sentence.

        Creates appropriate task descriptions with length management
        for BPMN documentation purposes.

        Args:
            sent_text: Source sentence for description

        Returns:
            Formatted task description
        """
        # Truncate long sentences for readability
        if len(sent_text) > 100:
            return f"{sent_text[:97]}..."
        return sent_text

    def _contains_timer_event(self, sent_text: str) -> bool:
        """
        Smart detection of timer events.

        Uses pattern matching to identify temporal constraints,
        deadlines, and timer-related elements in text.

        Args:
            sent_text: Text to analyze for timer indicators

        Returns:
            True if sentence contains timer event patterns
        """
        text_lower = sent_text.lower()
        for pattern in self.timer_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_timer_event(self, sent, section_header: str, context: List[str]) -> Optional[Dict]:
        """
        Extract timer event with enhanced timing details.

        Processes sentences containing timer patterns to extract detailed
        timing information and create timer event structures.

        Args:
            sent: spaCy sentence object containing timer information
            section_header: Section context for organization
            context: Previous sentences for actor resolution

        Returns:
            Timer event dictionary with timing metadata or None
        """
        sent_text = sent.text.strip()

        # Extract detailed timing information from sentence
        timing_info = self._extract_timing_information(sent_text)

        # Determine responsible actor using enhanced methods
        actor = self._extract_linguistic_actor(sent, context)
        if not actor:
            actor = self._create_smart_fallback_actor(section_header, sent_text)

        # Create timer name based on context analysis
        timer_name = self._determine_timer_type(sent_text)

        return {
            'type': 'timer_event',
            'actor': actor,
            'name': timer_name,
            'timing': timing_info,
            'description': f"Timer event: {sent_text[:80]}...",
            'section': section_header,
            'confidence': 0.9,
            'source_sentence': sent_text
        }

    def _extract_timing_information(self, sent_text: str) -> Dict[str, str]:
        """
        Extract detailed timing information from sentence.

        Processes temporal expressions to extract duration information,
        SLA requirements, and deadline constraints for timer events.

        Args:
            sent_text: Text containing timing information

        Returns:
            Dictionary with timing metadata and ISO duration formats
        """
        timing_info = {}

        # Pattern 1: Explicit durations with numeric values
        duration_patterns = [
            r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b',
            r'\bwithin\s+(\d+)\s*(minutes?|hours?|days?|weeks?)\b',
            r'\bafter\s+(\d+)\s*(minutes?|hours?|days?|weeks?)\b',
            r'\bevery\s+(\d+)\s*(minutes?|hours?|days?|weeks?)\b'
        ]

        for pattern in duration_patterns:
            match = re.search(pattern, sent_text.lower())
            if match:
                number = match.group(1)
                unit = match.group(2)

                # Normalize unit naming for consistency
                if unit.startswith('min'):
                    unit = 'minutes'
                elif unit.startswith('hr') or unit.startswith('hour'):
                    unit = 'hours'
                elif unit.startswith('day'):
                    unit = 'days'
                elif unit.startswith('week'):
                    unit = 'weeks'
                elif unit.startswith('month'):
                    unit = 'months'
                elif unit.startswith('year'):
                    unit = 'years'

                timing_info['duration'] = f"{number} {unit}"
                timing_info['iso_duration'] = self._convert_to_iso_duration(number, unit)
                break

        # Pattern 2: Relative time expressions
        if 'immediately' in sent_text.lower():
            timing_info['duration'] = 'immediate'
            timing_info['iso_duration'] = 'PT0S'
        elif 'real-time' in sent_text.lower() or 'realtime' in sent_text.lower():
            timing_info['duration'] = 'real-time'
            timing_info['iso_duration'] = 'PT0S'

        # Pattern 3: SLA and deadline indicators
        sla_patterns = [
            r'\bsla\s+(?:of\s+)?(\d+)\s*(minutes?|hours?|days?)\b',
            r'\bdeadline\s+(?:of\s+)?(\d+)\s*(minutes?|hours?|days?)\b',
            r'\btimeout\s+(?:after\s+)?(\d+)\s*(minutes?|hours?|days?)\b'
        ]

        for pattern in sla_patterns:
            match = re.search(pattern, sent_text.lower())
            if match:
                number = match.group(1)
                unit = match.group(2)
                timing_info['sla_duration'] = f"{number} {unit}"
                timing_info['type'] = 'deadline'
                break

        return timing_info

    def _convert_to_iso_duration(self, number: str, unit: str) -> str:
        """
        Convert duration to ISO 8601 format for BPMN.

        Transforms human-readable duration formats into ISO 8601
        duration strings for BPMN timer event configuration.

        Args:
            number: Numeric duration value
            unit: Time unit (minutes, hours, days, etc.)

        Returns:
            ISO 8601 duration string
        """
        try:
            num = int(number)

            if unit in ['minutes', 'mins']:
                return f"PT{num}M"
            elif unit in ['hours', 'hrs']:
                return f"PT{num}H"
            elif unit in ['days']:
                return f"P{num}D"
            elif unit in ['weeks']:
                return f"P{num}W"
            elif unit in ['months']:
                return f"P{num}M"
            elif unit in ['years']:
                return f"P{num}Y"
            else:
                return f"PT{num}M"  # Default to minutes for unknown units
        except ValueError:
            return "PT5M"  # Default fallback duration

    def _determine_timer_type(self, sent_text: str) -> str:
        """
        Determine the type of timer based on context.

        Analyzes sentence content to classify timer events into
        specific types based on their business purpose.

        Args:
            sent_text: Text containing timer information

        Returns:
            Descriptive timer type name
        """
        text_lower = sent_text.lower()

        if 'reminder' in text_lower or 'notify' in text_lower:
            return "Send Reminder"
        elif 'deadline' in text_lower or 'timeout' in text_lower:
            return "Deadline Monitor"
        elif 'sla' in text_lower:
            return "SLA Monitor"
        elif 'escalat' in text_lower:
            return "Escalation Timer"
        elif 'retry' in text_lower or 'attempt' in text_lower:
            return "Retry Timer"
        elif 'wait' in text_lower or 'delay' in text_lower:
            return "Wait Timer"
        else:
            return "Process Timer"

    def _contains_message_event(self, sent_text: str) -> bool:
        """
        Smart detection of message events.

        Uses pattern matching to identify communication flows,
        notifications, and message-based interactions in text.

        Args:
            sent_text: Text to analyze for message indicators

        Returns:
            True if sentence contains message event patterns
        """
        text_lower = sent_text.lower()
        for pattern in self.message_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_message_event(self, sent, section_header: str, context: List[str]) -> Optional[Dict]:
        """
        Extract message event with directional information.

        Processes sentences containing message patterns to identify
        communication direction and create message event structures.

        Args:
            sent: spaCy sentence object containing message information
            section_header: Section context for organization
            context: Previous sentences for actor resolution

        Returns:
            Message event dictionary with directional metadata or None
        """
        sent_text = sent.text.strip()

        # Determine message direction based on verb patterns
        is_sending = any(word in sent_text.lower() for word in ['send', 'transmit', 'notify'])
        is_receiving = any(word in sent_text.lower() for word in ['receive', 'callback', 'response'])

        actor = self._extract_linguistic_actor(sent, context) or "Message Handler"

        message_name = "Send Message" if is_sending else "Receive Message" if is_receiving else "Message Event"

        return {
            'type': 'message_event',
            'actor': actor,
            'name': message_name,
            'direction': 'outgoing' if is_sending else 'incoming' if is_receiving else 'intermediate',
            'description': f"Message event: {sent_text[:80]}...",
            'section': section_header,
            'confidence': 0.85,
            'source_sentence': sent_text
        }

    def _extract_linguistic_actor(self, sent, context: List[str]) -> Optional[str]:
        """
        Enhanced actor identification with better business patterns.

        Uses multiple linguistic analysis methods including business entity
        patterns, dependency parsing, named entity recognition, and context
        analysis to identify responsible actors.

        Args:
            sent: spaCy sentence object for linguistic analysis
            context: Previous sentences for context-based resolution

        Returns:
            Identified business actor name or None if not found
        """
        sent_text = sent.text.strip()

        # Method 1: Business entity pattern matching
        business_entity_patterns = [
            # Specific business entities and systems
            r'\b(Digital Channel|KYC Team|Core Banking|Risk Team|Compliance Team|Customer Service|Banking System|External Vendor)\b',
            r'\b(Customer|Applicant|Client|User)\b',
            r'\b([A-Z][a-z]+ (?:Team|Department|Lane|Channel|System|Engine|Service))\b',
            r'\b(Customer Success|Risk Engine|Banking Core|Digital Platform|Core System)\b',
            r'\bthe ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) (?:lane|team|department|system|engine)\b'
        ]

        for pattern in business_entity_patterns:
            matches = re.findall(pattern, sent_text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip() if isinstance(match, str) else match[0].strip()
                if len(clean_match) > 2 and self._is_business_actor_enhanced(clean_match):
                    return clean_match

        # Method 2: Dependency parsing for sentence subjects
        subjects = []
        for token in sent:
            if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ in ['NOUN', 'PROPN']:
                subject_phrase = self._build_noun_phrase(token)
                if self._is_business_actor_enhanced(subject_phrase):
                    subjects.append(subject_phrase)

        if subjects:
            return subjects[0]

        # Method 3: Named entity recognition for organizational entities
        doc = nlp(sent_text)
        for ent in doc.ents:
            if ent.label_ in ['ORG'] and self._is_business_actor_enhanced(ent.text):
                return ent.text

        # Method 4: Context-based resolution from previous sentences
        for sent_context in reversed(context[-2:]):
            for pattern in business_entity_patterns:
                matches = re.findall(pattern, sent_context, re.IGNORECASE)
                for match in matches:
                    clean_match = match.strip() if isinstance(match, str) else match[0].strip()
                    if len(clean_match) > 2 and self._is_business_actor_enhanced(clean_match):
                        return clean_match

        return None  # Return None to trigger smart fallback creation

    def _build_noun_phrase(self, token) -> str:
        """
        Build complete noun phrase from dependency tree.

        Constructs full noun phrases by collecting modifiers and
        compound elements from the dependency parse tree.

        Args:
            token: spaCy token representing the head noun

        Returns:
            Complete noun phrase string
        """
        phrase_tokens = [token]

        # Add modifiers and compound elements
        for child in token.children:
            if child.dep_ in ['compound', 'amod', 'nmod']:
                phrase_tokens.append(child)

        phrase_tokens.sort(key=lambda t: t.i)
        return ' '.join(t.text for t in phrase_tokens)

    def _is_business_actor_enhanced(self, actor: str) -> bool:
        """
        Enhanced business actor validation with comprehensive filtering.

        Validates actor candidates using expanded filtering for technical
        terms and enhanced business pattern recognition.

        Args:
            actor: String to validate as business actor

        Returns:
            True if string represents a valid business actor
        """
        if not actor or len(actor.strip()) < 2:
            return False

        actor_lower = actor.strip().lower()

        # Expanded technical terms and process artifacts filtering
        technical_terms = {
            'process', 'document', 'system', 'it', 'this', 'that', 'method', 'approach',
            'flow', 'event', 'task', 'step', 'action', 'gateway', 'timer', 'message',
            'comprehensive', 'narrative', 'parallel', 'exclusive', 'none', 'paths',
            'signed', 'bounced', 'conditional', 'synchronous', 'monitoring', 'receiving',
            # BPMN and technical terms
            'bpmn', 'json', 'xml', 'api', 'sla', 'oidc', 'utms', 'pdf', 'csv',
            'database', 'server', 'endpoint', 'workflow', 'subprocess', 'loop',
            'branch', 'merge', 'split', 'join', 'start', 'end', 'intermediate'
        }

        if actor_lower in technical_terms:
            return False

        # Filter single technical acronyms without business context
        if len(actor) <= 4 and actor.isupper() and actor_lower in technical_terms:
            return False

        # Strong business indicators with enhanced patterns
        strong_patterns = [
            # Organizational units
            r'team$', r'department$', r'lane$', r'channel$', r'engine$', r'service$',
            r'manager$', r'lead$', r'analyst$', r'officer$', r'committee$', r'unit$',

            # Business domains
            r'^(digital|core|risk|kyc|compliance|customer|banking|external|internal)',
            r'(success|support|validation|management|processing|operations)',
            r'(fraud|security|audit|regulatory|legal|finance)',

            # Specific business roles
            r'(specialist|coordinator|administrator|supervisor|director)'
        ]

        for pattern in strong_patterns:
            if re.search(pattern, actor_lower):
                return True

        # Multi-word business entities with domain validation
        if len(actor.split()) >= 2:
            words = actor_lower.split()

            # Business domain vocabulary
            business_words = {
                'digital', 'core', 'banking', 'kyc', 'risk', 'customer', 'compliance',
                'external', 'internal', 'fraud', 'security', 'audit', 'regulatory',
                'operations', 'finance', 'legal', 'marketing', 'sales', 'support'
            }

            # Must contain business words without technical terms
            has_business_word = any(word in business_words for word in words)
            has_technical_word = any(word in technical_terms for word in words)

            if has_business_word and not has_technical_word:
                return True

        return False

    def _extract_regular_tasks_enhanced(self, sent, section_header: str, context: List[str]) -> List[Dict]:
        """
        Enhanced task extraction with smart fallback logic.

        Processes sentences to extract business tasks using enhanced actor
        identification methods and smart fallback logic for actor assignment.

        Args:
            sent: spaCy sentence object for analysis
            section_header: Section context for organization
            context: Previous sentences for actor resolution

        Returns:
            List of enhanced task dictionaries
        """
        sent_text = sent.text.strip()
        tasks = []

        # Try enhanced actor extraction first
        actor = self._extract_linguistic_actor(sent, context)

        if not actor:
            # Fallback: try traditional method with enhanced filtering
            old_actors = self._extract_business_actors(sent_text)
            if old_actors:
                # Filter using enhanced validation
                filtered_actors = [a for a in old_actors if self._is_business_actor_enhanced(a)]
                actor = filtered_actors[0] if filtered_actors else None

        if not actor:
            # Smart fallback: create actor based on section and sentence content
            actor = self._create_smart_fallback_actor(section_header, sent_text)

        # Extract business actions using existing method
        actions = self._extract_business_actions(sent)
        if not actions:
            return tasks

        primary_action = actions[0]
        task_name = self._create_task_name(primary_action, sent_text)

        task = {
            'type': 'task',
            'actor': actor,
            'name': task_name,
            'description': self._create_task_description(sent_text),
            'section': section_header,
            'confidence': self._calculate_confidence_enhanced(actor, primary_action, sent_text),
            'source_sentence': sent_text
        }

        tasks.append(task)
        return tasks

    def _create_smart_fallback_actor(self, section_header: str, sent_text: str) -> str:
        """
        Create intelligent fallback actors based on context analysis.

        Uses section headers and sentence content to assign appropriate
        business actors when direct identification methods fail.

        Args:
            section_header: Section title for domain context
            sent_text: Sentence content for activity context

        Returns:
            Contextually appropriate fallback actor name
        """
        section_lower = section_header.lower()
        sent_lower = sent_text.lower()

        # Banking domain assignments based on section context
        if any(word in section_lower for word in ['kyc', 'know your customer', 'compliance', 'due diligence']):
            return "KYC Team"
        elif any(word in section_lower for word in ['risk', 'fraud', 'security']):
            return "Risk Team"
        elif any(word in section_lower for word in ['digital', 'online', 'web', 'portal', 'channel']):
            return "Digital Channel"
        elif any(word in section_lower for word in ['banking', 'core', 'account', 'transaction']):
            return "Core Banking"
        elif any(word in section_lower for word in ['customer', 'client', 'user']):
            return "Customer Service"

        # Sentence-level activity assignments
        if any(word in sent_lower for word in ['validate', 'verify', 'check', 'review']):
            return "Validation Team"
        elif any(word in sent_lower for word in ['approve', 'reject', 'decide']):
            return "Approval Team"
        elif any(word in sent_lower for word in ['monitor', 'track', 'observe']):
            return "Monitoring Team"
        elif any(word in sent_lower for word in ['generate', 'create', 'build']):
            return "Processing Team"

        # Default fallback for unclassifiable contexts
        return "Process Owner"

    def _calculate_confidence_enhanced(self, actor: str, action: str, sent_text: str) -> float:
        """
        Enhanced confidence calculation with multiple factors.

        Calculates task confidence scores based on actor quality,
        action specificity, and sentence characteristics.

        Args:
            actor: Identified actor name
            action: Extracted business action
            sent_text: Source sentence text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5

        # Higher confidence for enhanced actor detection
        if self._is_business_actor_enhanced(actor):
            confidence += 0.3  # Significant boost for linguistic detection

        # Higher confidence for recognized business verbs
        if any(verb in action.lower() for verb in self.business_verbs):
            confidence += 0.2

        # Sentence specificity and detail level
        if len(sent_text.split()) > 10:
            confidence += 0.1

        return min(confidence, 1.0)

    def _merge_message_decision_triplets(self, workflow_tasks: List[Dict]) -> List[Dict]:
        """
        Merge logical triplets of send-message/receive-message/decision.

        Identifies and merges related sequences of message sending, message
        receiving, and decision-making into cohesive workflow elements.

        Args:
            workflow_tasks: List of workflow tasks to analyze for triplets

        Returns:
            List with merged triplets and unchanged individual tasks
        """
        if len(workflow_tasks) < 3:
            return workflow_tasks

        merged_tasks = []
        i = 0

        while i < len(workflow_tasks):
            current = workflow_tasks[i]

            # Look for message triplet pattern: SEND → RECEIVE → DECISION
            if (i + 2 < len(workflow_tasks) and
                    current['type'] == 'message_event' and current.get('direction') == 'outgoing' and
                    workflow_tasks[i + 1]['type'] == 'message_event' and workflow_tasks[i + 1].get(
                        'direction') == 'incoming' and
                    workflow_tasks[i + 2]['type'] == 'gateway'):

                send_task = current
                receive_task = workflow_tasks[i + 1]
                decision_task = workflow_tasks[i + 2]

                # Create merged exchange task with combined information
                merged_task = {
                    'type': 'message_event',
                    'actor': send_task['actor'],
                    'name': f"Exchange: {send_task['name']} / {receive_task['name']}",
                    'direction': 'exchange',
                    'description': f"Message exchange ending with decision: {decision_task['name']}",
                    'section': send_task['section'],
                    'confidence': (send_task.get('confidence', 0.5) + receive_task.get('confidence',
                                                                                       0.5) + decision_task.get(
                        'confidence', 0.5)) / 3,
                    'source_sentence': send_task.get('source_sentence', ''),
                    'branches': decision_task.get('branches', []),
                    'merged_from': ['send', 'receive', 'decision']
                }

                merged_tasks.append(merged_task)
                i += 3  # Skip all three tasks in the triplet
                print(f"MERGED TRIPLET: {send_task['name']} + {receive_task['name']} + {decision_task['name']}")

            else:
                merged_tasks.append(current)
                i += 1

        return merged_tasks